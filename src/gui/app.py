import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
from src.utils.logger import app_logger
from src.gui.filter_tab import FilterTab
from src.gui.analysis_tab import AnalysisTab
from src.gui.view_tab import ViewTab
from src.gui.action_potential_tab import ActionPotentialTab
from src.io_utils.io_utils import ATFHandler
from src.filtering.filtering import combined_filter
from src.analysis.action_potential import ActionPotentialProcessor
from src.gui.window_manager import SignalWindowManager

class SignalAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Signal Analyzer")
        
        # Initialize data variables
        self.data = None
        self.time_data = None
        self.filtered_data = None
        self.current_filters = {}
        self.action_potential_processor = None
        
        # Create main layout
        self.setup_main_layout()
        
        # Initialize window manager BEFORE toolbar setup
        self.window_manager = SignalWindowManager(self.master)
        
        # Setup components
        self.setup_toolbar()
        self.setup_plot()
        self.setup_tabs()
        
        app_logger.info("Application initialized successfully")

    def setup_main_layout(self):
        """Setup the main application layout"""
        # Create main frames
        self.toolbar_frame = ttk.Frame(self.master)
        self.toolbar_frame.pack(fill='x', padx=5, pady=5)
        
        # Create main container for plot and controls
        self.main_container = ttk.PanedWindow(self.master, orient='horizontal')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create frames for plot and controls
        self.plot_frame = ttk.Frame(self.main_container)
        self.control_frame = ttk.Frame(self.main_container)
        
        # Add frames to PanedWindow
        self.main_container.add(self.plot_frame, weight=3)
        self.main_container.add(self.control_frame, weight=1)

    def setup_toolbar(self):
        """Setup the toolbar with file operations"""
        # File operations
        file_frame = ttk.Frame(self.toolbar_frame)
        file_frame.pack(side='left', fill='x')
        
        ttk.Button(file_frame, text="Load Data", 
                  command=self.load_data).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Export Data", 
                  command=self.export_data).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Export Figure", 
                  command=self.export_figure).pack(side='left', padx=2)
        
        # Add separator
        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        
        # Add Separate Plots button
        plots_frame = ttk.Frame(self.toolbar_frame)
        plots_frame.pack(side='left', fill='x')
        
        self.separate_plots_btn = ttk.Button(
            plots_frame, 
            text="Separate Plots ▼",
            command=self.show_plot_menu
        )
        self.separate_plots_btn.pack(side='left', padx=2)
        
        # Create plot selection menu
        self.plot_menu = tk.Menu(self.master, tearoff=0)
        self.plot_menu.add_command(
            label="Baseline Correction",
            command=lambda: self.window_manager.open_baseline_window(preserve_main=True)
        )
        self.plot_menu.add_command(
            label="Normalization",
            command=lambda: self.window_manager.open_normalization_window(preserve_main=True)
        )
        self.plot_menu.add_command(
            label="Integration",
            command=lambda: self.window_manager.open_integration_window(preserve_main=True)
        )
        self.plot_menu.add_separator()
        self.plot_menu.add_command(
            label="Close All Plot Windows",
            command=self.window_manager.close_all_windows
        )
        
        # Status label
        self.status_var = tk.StringVar(value="No data loaded")
        self.status_label = ttk.Label(self.toolbar_frame, 
                                    textvariable=self.status_var)
        self.status_label.pack(side='right', padx=5)

    def show_plot_menu(self, event=None):
        """Show the plot selection menu below the Separate Plots button"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Get button position
        x = self.separate_plots_btn.winfo_rootx()
        y = self.separate_plots_btn.winfo_rooty() + self.separate_plots_btn.winfo_height()
        
        # Show menu at button position
        self.plot_menu.post(x, y)

    def setup_plot(self):
        """Setup the matplotlib plot area"""
        # Create figure and canvas
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas and navigation toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def setup_tabs(self):
        """Setup the control tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.filter_tab = FilterTab(self.notebook, self.on_filter_change)
        self.analysis_tab = AnalysisTab(self.notebook, self.on_analysis_update)
        self.view_tab = ViewTab(self.notebook, self.on_view_change)
        self.action_potential_tab = ActionPotentialTab(self.notebook, self.on_action_potential_analysis)
        
        # Add tabs to notebook
        self.notebook.add(self.filter_tab.frame, text='Filters')
        self.notebook.add(self.analysis_tab.frame, text='Analysis')
        self.notebook.add(self.view_tab.frame, text='View')
        self.notebook.add(self.action_potential_tab.frame, text='Action Potential')

    def load_data(self):
        """Load data from file"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return
                
            app_logger.info(f"Loading file: {filepath}")
            
            # Load ATF file
            atf_handler = ATFHandler(filepath)
            atf_handler.load_atf()
            
            # Get data
            self.time_data = atf_handler.get_column("Time")
            self.data = atf_handler.get_column("#1")
            self.filtered_data = self.data.copy()
            
            # Update window manager
            self.window_manager.set_data(self.time_data, self.data)
            
            # Update view limits
            self.view_tab.update_limits(
                t_min=self.time_data[0],
                t_max=self.time_data[-1],
                v_min=np.min(self.data),
                v_max=np.max(self.data)
            )
            
            # Update plot
            self.update_plot()
            self.status_var.set(f"Loaded: {filepath.split('/')[-1]}")
            
            # Update analysis
            self.analysis_tab.update_data(self.data, self.time_data)
            
            # Reset action potential processor
            self.action_potential_processor = None
            
            app_logger.info("Data loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Error loading data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def on_filter_change(self, filters):
        """Handle changes in filter settings"""
        if self.data is None:
            return
            
        try:
            app_logger.debug("Applying filters with parameters: " + str(filters))
            self.current_filters = filters
            
            # Apply filters
            self.filtered_data = combined_filter(self.data, **filters)
            
            # Update plot and analysis
            self.update_plot()
            self.analysis_tab.update_filtered_data(self.filtered_data)
            
            # Update window manager
            self.window_manager.set_data(self.time_data, self.filtered_data)
            
            # Reset action potential processor when filters change
            self.action_potential_processor = None
            
        except Exception as e:
            app_logger.error(f"Error applying filters: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply filters: {str(e)}")

    def on_analysis_update(self, analysis_params):
        """Handle changes in analysis settings"""
        if self.filtered_data is None:
            return
            
        try:
            # Update analysis display
            self.analysis_tab.analyze_data(
                self.filtered_data, 
                self.time_data,
                analysis_params
            )
        except Exception as e:
            app_logger.error(f"Error updating analysis: {str(e)}")

    def on_action_potential_analysis(self, params):
        """Handle action potential analysis"""
        try:
            if self.filtered_data is None:
                messagebox.showwarning("Analysis", "No filtered data available")
                return None

            # Create processor with current data
            processor = ActionPotentialProcessor(self.filtered_data, self.time_data, params)
            
            # Process signal and get results
            processed_data, time_data, results = processor.process_signal()
            
            # Update plot if successful
            if processed_data is not None and results:
                self.plot_action_potential(processed_data, time_data)
                app_logger.info("Action potential analysis completed successfully")
                return results
                
            return None
                
        except Exception as e:
            app_logger.error(f"Error in action potential analysis: {str(e)}")
            raise

    def plot_action_potential(self, processed_data, time_data):
        """Plot action potential analysis results"""
        try:
            self.ax.clear()
            
            # Plot original signal with transparency
            self.ax.plot(self.time_data, self.data, 'b-', alpha=0.3, label='Original')
            
            # Plot filtered signal
            if self.filtered_data is not None:
                self.ax.plot(self.time_data, self.filtered_data, 'r-', alpha=0.5, label='Filtered')
            
            # Plot processed data
            self.ax.plot(time_data, processed_data, 'g-', linewidth=2, label='Processed')
            
            # Set labels and grid
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True)
            self.ax.legend()
            
            # Update display
            self.fig.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as e:
            app_logger.error(f"Error plotting action potential: {str(e)}")
            raise

    def on_view_change(self, view_params):
        """Handle changes in view settings"""
        if self.data is None:
            return
            
        try:
            self.update_plot(view_params)
        except Exception as e:
            app_logger.error(f"Error updating view: {str(e)}")

    def update_plot(self, view_params=None):
        """Update the plot with current data and view settings"""
        if self.data is None:
            return
            
        try:
            self.ax.clear()
            
            # Get view parameters
            if view_params is None:
                view_params = self.view_tab.get_view_params()
            
            # Get plot range
            if view_params.get('use_interval', False):
                start_idx = np.searchsorted(self.time_data, view_params['t_min'])
                end_idx = np.searchsorted(self.time_data, view_params['t_max'])
                plot_time = self.time_data[start_idx:end_idx]
                plot_data = self.data[start_idx:end_idx]
                if self.filtered_data is not None:
                    plot_filtered = self.filtered_data[start_idx:end_idx]
            else:
                plot_time = self.time_data
                plot_data = self.data
                plot_filtered = self.filtered_data
            
            # Plot data
            if view_params.get('show_original', True):
                self.ax.plot(plot_time, plot_data, 'b-', 
                           label='Original Signal', alpha=0.5)
            
            if view_params.get('show_filtered', True) and plot_filtered is not None:
                self.ax.plot(plot_time, plot_filtered, 'r-', 
                           label='Filtered Signal')
            
            # Set labels and grid
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True)
            self.ax.legend()
            
            # Update axis limits if specified
            if 'y_min' in view_params and 'y_max' in view_params:
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            
            self.fig.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as e:
            app_logger.error(f"Error updating plot: {str(e)}")
            raise

    def export_data(self):
        """Export the current data to CSV"""
        if self.filtered_data is None:
            messagebox.showwarning("Export", "No filtered data to export")
            return
            
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filepath:
                df = pd.DataFrame({
                    'Time': self.time_data,
                    'Original': self.data,
                    'Filtered': self.filtered_data
                })
                
                # Add action potential analysis results if available
                if self.action_potential_processor is not None:
                    df['Processed'] = self.filtered_data
                
                df.to_csv(filepath, index=False)
                
                app_logger.info(f"Data exported to {filepath}")
                messagebox.showinfo("Export", "Data exported successfully")
                
        except Exception as e:
            app_logger.error(f"Error exporting data: {str(e)}")
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def export_figure(self):
        """Export the current figure"""
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if filepath:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                app_logger.info(f"Figure exported to {filepath}")
                messagebox.showinfo("Export", "Figure exported successfully")
                
        except Exception as e:
            app_logger.error(f"Error exporting figure: {str(e)}")
            messagebox.showerror("Error", f"Failed to export figure: {str(e)}")

    def update_plot_with_processed_data(self, processed_data, processed_time):
        """Update plot with processed data ensuring time alignment"""
        try:
            self.ax.clear()
            
            # Get view parameters
            view_params = self.view_tab.get_view_params()
            
            # Plot original data with transparency
            if view_params.get('show_original', True):
                self.ax.plot(self.time_data, self.data, 'b-', 
                           label='Original Signal', alpha=0.3)
            
            # Plot filtered data
            if view_params.get('show_filtered', True):
                self.ax.plot(self.time_data, self.filtered_data, 'r-', 
                           label='Filtered Signal', alpha=0.5)
            
            # Plot processed data
            self.ax.plot(processed_time, processed_data, 'g-', 
                        label='Processed Signal', linewidth=2)
            
            # Set labels and grid
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True)
            self.ax.legend()
            
            # Update axis limits if specified
            if 'y_min' in view_params and 'y_max' in view_params:
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            
            self.fig.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as e:
            app_logger.error(f"Error updating plot with processed data: {str(e)}")
            raise

# For standalone testing
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SignalAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        app_logger.critical(f"Application crashed: {str(e)}")
        raise