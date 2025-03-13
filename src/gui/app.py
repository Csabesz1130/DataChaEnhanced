import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import app_logger
from src.gui.filter_tab import FilterTab
from src.gui.analysis_tab import AnalysisTab
from src.gui.view_tab import ViewTab
from src.gui.action_potential_tab import ActionPotentialTab
from src.io_utils.io_utils import ATFHandler
from src.filtering.filtering import combined_filter
from src.analysis.action_potential import ActionPotentialProcessor
from src.gui.window_manager import SignalWindowManager
from src.utils.analysis_history_manager import AnalysisHistoryManager
from src.gui.history_window import HistoryWindow
from src.excel_export import add_excel_export_to_app

class SignalAnalyzerApp:
    def __init__(self, master):
        """Initialize the Signal Analyzer application."""
        self.master = master
        self.master.title("Signal Analyzer")
        
        # Initialize data variables
        self.data = None
        self.time_data = None
        self.filtered_data = None
        self.current_filters = {}
        self.action_potential_processor = None
        self.current_file = None  # Track current file path

        self.use_regression_hyperpol = tk.BooleanVar(value=False)
        self.use_regression_depol = tk.BooleanVar(value=False)
        
        # Create main layout
        self.setup_main_layout()
        
        # Initialize window manager BEFORE toolbar setup
        self.window_manager = SignalWindowManager(self.master)
        
        # Initialize history manager
        self.history_manager = AnalysisHistoryManager(self)
        
        # Setup components
        self.setup_toolbar()
        self.setup_plot()
        self.setup_plot_interaction()
        self.setup_tabs()

        add_excel_export_to_app(self)
        
        app_logger.info("Application initialized successfully")

    def setup_main_layout(self):
        """Setup the main application layout with resizable panes"""
        self.toolbar_frame = ttk.Frame(self.master)
        self.toolbar_frame.pack(fill='x', padx=5, pady=5)
        
        self.main_container = ttk.PanedWindow(self.master, orient='horizontal')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        total_width = self.master.winfo_screenwidth()
        plot_width = int(total_width * 0.7)
        control_width = int(total_width * 0.3)
        
        self.plot_frame = ttk.Frame(self.main_container, width=plot_width)
        self.control_frame = ttk.Frame(self.main_container, width=control_width)
        
        self.main_container.add(self.plot_frame, weight=70)
        self.main_container.add(self.control_frame, weight=30)
        
        self.plot_frame.pack_propagate(False)
        self.control_frame.pack_propagate(False)
        
        self.plot_frame.configure(width=400)
        self.control_frame.configure(width=300)
        
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill='both', expand=True)

    def setup_toolbar(self):
        """Setup the toolbar with file operations"""
        file_frame = ttk.Frame(self.toolbar_frame)
        file_frame.pack(side='left', fill='x')
        
        ttk.Button(file_frame, text="Load Data", 
                command=self.load_data).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Export Data", 
                command=self.export_data).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Export Figure", 
                command=self.export_figure).pack(side='left', padx=2)
        
        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side='left', fill='y', padx=5)

        # Export curves and view history buttons
        export_frame = ttk.Frame(self.toolbar_frame)
        export_frame.pack(side='left', padx=5)

        self.export_curves_btn = ttk.Button(
            export_frame,
            text="Export Purple Curves",
            command=self.on_export_purple_curves
        )
        self.export_curves_btn.pack(side='left', padx=2)
        
        # Add View History button
        ttk.Button(
            export_frame,
            text="View History",
            command=self.show_analysis_history
        ).pack(side='left', padx=2)

        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        
        plots_frame = ttk.Frame(self.toolbar_frame)
        plots_frame.pack(side='left', fill='x')
        
        self.separate_plots_btn = ttk.Button(
            plots_frame, 
            text="Separate Plots ▼",
            command=self.show_plot_menu
        )
        self.separate_plots_btn.pack(side='left', padx=2)
        
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
        
        self.status_var = tk.StringVar(value="No data loaded")
        self.status_label = ttk.Label(self.toolbar_frame, textvariable=self.status_var)
        self.status_label.pack(side='right', padx=5)

    def show_plot_menu(self, event=None):
        """Show the plot selection menu below the Separate Plots button"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        x = self.separate_plots_btn.winfo_rootx()
        y = self.separate_plots_btn.winfo_rooty() + self.separate_plots_btn.winfo_height()
        
        self.plot_menu.post(x, y)

    def show_analysis_history(self):
        """Show the analysis history dialog."""
        if not hasattr(self, 'history_manager'):
            messagebox.showinfo("History", "History manager not initialized.")
            return
            
        if not self.history_manager.history_entries:
            messagebox.showinfo("History", "No analysis history available.")
            return
            
        try:
            # Import the HistoryWindow class directly here in case of import issues
            from src.gui.history_window import HistoryWindow
            # Create history window
            history_window = HistoryWindow(self.master, self.history_manager)
        except Exception as e:
            app_logger.error(f"Error showing history window: {str(e)}")
            messagebox.showerror("Error", f"Failed to show history: {str(e)}")

    def show_history(self):
        """Show the analysis history window."""
        try:
            history_window = HistoryWindow(self.master, self.history_manager)
            history_window.transient(self.master)
            history_window.grab_set()
        except Exception as e:
            app_logger.error(f"Error showing history window: {str(e)}")
            messagebox.showerror("Error", f"Failed to show history: {str(e)}")

    def setup_plot_interaction(self):
        """Setup interactive plot selection and regression visualization."""
        from matplotlib.widgets import SpanSelector
        
        def on_select(xmin, xmax):
            """Handle region selection on plot."""
            if not hasattr(self, 'action_potential_tab'):
                return
            
            try:
                if not hasattr(self.action_potential_processor, 'modified_hyperpol_times'):
                    return
                    
                # Convert milliseconds to seconds for calculation
                xmin_sec = xmin / 1000
                xmax_sec = xmax / 1000
                
                # Get time range in seconds
                time_range = (self.action_potential_processor.modified_hyperpol_times[-1] - 
                            self.action_potential_processor.modified_hyperpol_times[0])
                
                # Calculate indices (0-199 range)
                start_idx = int((xmin_sec - self.action_potential_processor.modified_hyperpol_times[0]) / 
                            time_range * 199)
                end_idx = int((xmax_sec - self.action_potential_processor.modified_hyperpol_times[0]) / 
                            time_range * 199)
                
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, 198))
                end_idx = max(1, min(end_idx, 199))
                
                # Update sliders
                if hasattr(self.action_potential_tab, 'regression_start'):
                    self.action_potential_tab.regression_start.set(start_idx)
                if hasattr(self.action_potential_tab, 'regression_end'):
                    self.action_potential_tab.regression_end.set(end_idx)
                
                # Update display
                if hasattr(self.action_potential_tab, 'on_regression_interval_change'):
                    self.action_potential_tab.on_regression_interval_change()
                app_logger.debug(f"Selected range: {start_idx}-{end_idx}")
                
            except Exception as e:
                app_logger.error(f"Error in span selection: {str(e)}")
        
        # Create span selector with better visibility
        self.span_selector = SpanSelector(
            self.ax,
            on_select,
            'horizontal',
            useblit=True,
            props=dict(
                alpha=0.3,
                facecolor='lightblue',
                edgecolor='blue'
            ),
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Initially hide it
        self.span_selector.set_visible(False)
        
        # Setup interactive range span selectors
        self.setup_interactive_ranges()

    def toggle_span_selectors(self, visible):
        """Toggle the visibility and activity of all span selectors."""
        if hasattr(self, 'span_selector'):
            try:
                # Get current display mode
                display_mode = "line"
                if hasattr(self.action_potential_tab, 'modified_display_mode'):
                    display_mode = self.action_potential_tab.modified_display_mode.get()
                
                # Enable if visible is True and display mode is either line or all_points
                active = visible and display_mode in ["line", "all_points"]
                
                # Set visibility for the main span selector (regression)
                self.span_selector.set_visible(active)
                self.span_selector.set_active(active)
                
                # Also toggle the range span selectors if they exist
                if hasattr(self, 'hyperpol_span'):
                    self.hyperpol_span.set_visible(active)
                    self.hyperpol_span.set_active(active)
                    
                if hasattr(self, 'depol_span'):
                    self.depol_span.set_visible(active)
                    self.depol_span.set_active(active)
                
                # Ensure the canvas is updated
                self.canvas.draw_idle()
                
                app_logger.debug(f"All span selectors visibility set to {active} (requested: {visible}, mode: {display_mode})")
                
            except Exception as e:
                app_logger.error(f"Error toggling span selectors: {str(e)}")
                # Ensure selectors are hidden on error
                self.span_selector.set_visible(False)
                self.span_selector.set_active(False)
                
                if hasattr(self, 'hyperpol_span'):
                    self.hyperpol_span.set_visible(False)
                    self.hyperpol_span.set_active(False)
                    
                if hasattr(self, 'depol_span'):
                    self.depol_span.set_visible(False)
                    self.depol_span.set_active(False)
                    
                self.canvas.draw_idle()

    def plot_regression_lines(self, data, times, interval, color='purple', alpha=0.7):
        """Plot regression line for a given segment of data with improved visibility."""
        if data is None or times is None or len(data) < 2:
            return

        try:
            start_idx, end_idx = interval
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(data):
                return

            # Extract data for regression
            x = times[start_idx:end_idx]
            y = data[start_idx:end_idx]
            
            # Scatter points in selection range
            self.ax.scatter(x * 1000, y, 
                        color=color, 
                        alpha=alpha,
                        s=30,  # Increased point size
                        zorder=5)  # Ensure points are on top

            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            reg_line = np.poly1d(coeffs)

            # Calculate R-squared
            y_pred = reg_line(x)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

            # Plot the regression line
            self.ax.plot(x * 1000, reg_line(x), 
                        '--',  # Dashed line
                        color=color, 
                        alpha=alpha,
                        linewidth=2,
                        zorder=6,  # Make sure line is above points
                        label=f'Fit (R²={r2:.3f})')

            # Add slope annotation
            slope = coeffs[0]
            slope_text = f'Slope: {slope:.2f} pA/s'
            mid_x = np.mean(x) * 1000
            mid_y = np.mean(y)
            self.ax.annotate(slope_text, 
                            xy=(mid_x, mid_y),
                            xytext=(10, 10), 
                            textcoords='offset points',
                            color=color,
                            alpha=alpha,
                            bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))

            app_logger.debug(f"Regression line plotted: {slope_text}, R²={r2:.3f}")

        except Exception as e:
            app_logger.error(f"Error plotting regression line: {str(e)}")

    def setup_plot(self):
        """Setup the matplotlib plot area with cursor tracking."""
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        self.cursor_label = ttk.Label(self.plot_frame, text="")
        self.cursor_label.pack(side='bottom', fill='x')
        
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def setup_tabs(self):
        """Setup the control tabs"""
        self.filter_tab = FilterTab(self.notebook, self.on_filter_change)
        self.analysis_tab = AnalysisTab(self.notebook, self.on_analysis_update)
        self.view_tab = ViewTab(self.notebook, self.on_view_change)
        self.action_potential_tab = ActionPotentialTab(self.notebook, self.on_action_potential_analysis)
        
        self.notebook.add(self.filter_tab.frame, text='Filters')
        self.notebook.add(self.analysis_tab.frame, text='Analysis')
        self.notebook.add(self.view_tab.frame, text='View')
        self.notebook.add(self.action_potential_tab.frame, text='Action Potential')


    def _find_closest_point(self, x_ms, y_pA):
        """Find closest orange point when in All Points mode."""
        if (not hasattr(self, 'orange_curve_times') or 
            self.orange_curve_times is None or 
            not hasattr(self.action_potential_tab, 'average_display_mode') or
            self.action_potential_tab.average_display_mode.get() != "all_points"):
            return None, None
                
        # Convert x_ms from milliseconds to seconds for comparison
        x_sec = x_ms / 1000.0
        
        # Find closest point index
        diffs = np.abs(self.orange_curve_times - x_sec)
        closest_idx = np.argmin(diffs)
        
        # Use 0.0002 seconds threshold (0.2ms)
        if diffs[closest_idx] > 0.0002:
            return None, None
            
        # Check amplitude is close enough (within 20 pA)
        if abs(self.orange_curve[closest_idx] - y_pA) > 20:
            return None, None
            
        point_number = closest_idx + 1
        return point_number, self.orange_curve_times[closest_idx]

    def on_mouse_move(self, event):
        """Handle cursor movement and point detection."""
        if event.inaxes != self.ax:
            self.cursor_label.config(text="")
            return

        if self.time_data is None:
            return
        
        cursor_text = f"Time: {event.xdata:.1f} ms, Current: {event.ydata:.1f} pA"
        
        # Only find points if we have data and are in All Points mode
        if (hasattr(self, 'orange_curve') and 
            hasattr(self.action_potential_tab, 'average_display_mode') and
            self.action_potential_tab.average_display_mode.get() == "all_points"):
            
            point_num, _ = self._find_closest_point(event.xdata, event.ydata)
            if point_num is not None:
                cursor_text += f", Orange Point: {point_num}"

        self.cursor_label.config(text=cursor_text)

    def on_export_purple_curves(self):
        """
        Handler for the 'Export Purple Curves' button.
        1) Grab the processor + results from your analysis
        2) Ask user for a save filename
        3) Call processor.export_all_curves(...)
        """
        # Make sure we have a valid processor
        if not self.action_potential_processor:
            messagebox.showwarning("No Data", "Please run analysis before exporting.")
            return

        # If you store the final integrals in self.action_potential_tab somewhere:
        # or in a local variable from last analysis:
        results_dict = self.action_potential_tab.get_results_dict()  # or whatever your code uses
        if not results_dict:
            messagebox.showinfo("No Results", "No integral results found yet.")
            return

        # Prompt for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not save_path:
            return  # user canceled

        # Now call the function in the processor
        try:
            self.action_potential_processor.export_all_curves(results_dict, save_path)
            messagebox.showinfo("Export", f"Exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def setup_interactive_ranges(self):
        """Setup interactive draggable span selectors for integration ranges."""
        from matplotlib.widgets import SpanSelector
        
        # Create span selector for hyperpolarization range with reduced sensitivity
        self.hyperpol_span = SpanSelector(
            self.ax,
            self.on_hyperpol_span_select,
            'horizontal',
            useblit=True,
            props=dict(
                alpha=0.3,
                facecolor='blue',
                edgecolor='blue',
                linewidth=2  # Thicker border for better visibility
            ),
            interactive=True,
            drag_from_anywhere=True,
            button=1,        # Only respond to left mouse button
            minspan=10,      # Minimum span size in pixels - prevents tiny selections
            grab_range=8     # Makes it easier to grab the span (pixels)
        )
        
        # Create span selector for depolarization range with reduced sensitivity
        self.depol_span = SpanSelector(
            self.ax,
            self.on_depol_span_select,
            'horizontal',
            useblit=True,
            props=dict(
                alpha=0.3,
                facecolor='red',
                edgecolor='red',
                linewidth=2  # Thicker border for better visibility
            ),
            interactive=True,
            drag_from_anywhere=True,
            button=1,        # Only respond to left mouse button
            minspan=10,      # Minimum span size in pixels - prevents tiny selections
            grab_range=8     # Makes it easier to grab the span (pixels)
        )
        
        # Initially hide them
        self.hyperpol_span.set_visible(False)
        self.depol_span.set_visible(False)
        
        # Store previous extents to prevent unwanted movement
        self.prev_hyperpol_extents = None
        self.prev_depol_extents = None
        
        app_logger.debug("Interactive integration ranges initialized with improved sensitivity")

    def on_depol_span_select(self, xmin, xmax):
        """Handle dragging of the depolarization range with point snapping."""
        if not hasattr(self, 'action_potential_processor') or self.action_potential_processor is None:
            # Try using direct processor reference
            if not hasattr(self.action_potential_tab, 'processor') or self.action_potential_tab.processor is None:
                return
            processor = self.action_potential_tab.processor
        else:
            processor = self.action_potential_processor
            
        try:
            if not hasattr(processor, 'modified_depol_times'):
                return
                
            # Get depol times array for index calculation
            depol_times = processor.modified_depol_times
            if depol_times is None or len(depol_times) == 0:
                return
                
            # Convert milliseconds to seconds for calculation
            xmin_sec = xmin / 1000
            xmax_sec = xmax / 1000
            
            # Find the closest actual data points rather than interpolating
            # This creates a "snapping" effect to the real data points
            start_idx = 0
            end_idx = len(depol_times) - 1
            
            # Find closest point to xmin
            min_dist = float('inf')
            for i, t in enumerate(depol_times):
                dist = abs(t - xmin_sec)
                if dist < min_dist:
                    min_dist = dist
                    start_idx = i
                    
            # Find closest point to xmax
            min_dist = float('inf')
            for i, t in enumerate(depol_times):
                dist = abs(t - xmax_sec)
                if dist < min_dist:
                    min_dist = dist
                    end_idx = i
            
            # Ensure start_idx < end_idx and at least 1 point apart
            if start_idx >= end_idx:
                if start_idx == len(depol_times) - 1:
                    start_idx = max(0, len(depol_times) - 2)
                else:
                    end_idx = min(start_idx + 1, len(depol_times) - 1)
            
            # Update the span extents to the actual data point times
            # This makes the span appear to "snap" to data points
            self.depol_span.extents = (
                depol_times[start_idx] * 1000, 
                depol_times[end_idx] * 1000
            )
            
            # Update the sliders in the action potential tab
            if hasattr(self.action_potential_tab, 'depol_start'):
                self.action_potential_tab.depol_start.set(start_idx)
            if hasattr(self.action_potential_tab, 'depol_end'):
                self.action_potential_tab.depol_end.set(end_idx)
            
            # Trigger the interval change handler
            if hasattr(self.action_potential_tab, 'on_integration_interval_change'):
                self.action_potential_tab.on_integration_interval_change()
            
            app_logger.debug(f"Depol range snapped to points: {start_idx}-{end_idx}")
            
        except Exception as e:
            app_logger.error(f"Error updating depol range: {str(e)}")

    def on_hyperpol_span_select(self, xmin, xmax):
        """Handle dragging of the hyperpolarization range with point snapping."""
        if not hasattr(self, 'action_potential_processor') or self.action_potential_processor is None:
            # Try using direct processor reference
            if not hasattr(self.action_potential_tab, 'processor') or self.action_potential_tab.processor is None:
                return
            processor = self.action_potential_tab.processor
        else:
            processor = self.action_potential_processor
            
        try:
            if not hasattr(processor, 'modified_hyperpol_times'):
                return
                
            # Get hyperpol times array for index calculation
            hyperpol_times = processor.modified_hyperpol_times
            if hyperpol_times is None or len(hyperpol_times) == 0:
                return
                
            # Convert milliseconds to seconds for calculation
            xmin_sec = xmin / 1000
            xmax_sec = xmax / 1000
            
            # Find the closest actual data points rather than interpolating
            # This creates a "snapping" effect to the real data points
            start_idx = 0
            end_idx = len(hyperpol_times) - 1
            
            # Find closest point to xmin
            min_dist = float('inf')
            for i, t in enumerate(hyperpol_times):
                dist = abs(t - xmin_sec)
                if dist < min_dist:
                    min_dist = dist
                    start_idx = i
                    
            # Find closest point to xmax
            min_dist = float('inf')
            for i, t in enumerate(hyperpol_times):
                dist = abs(t - xmax_sec)
                if dist < min_dist:
                    min_dist = dist
                    end_idx = i
            
            # Ensure start_idx < end_idx and at least 1 point apart
            if start_idx >= end_idx:
                if start_idx == len(hyperpol_times) - 1:
                    start_idx = max(0, len(hyperpol_times) - 2)
                else:
                    end_idx = min(start_idx + 1, len(hyperpol_times) - 1)
            
            # Update the span extents to the actual data point times
            # This makes the span appear to "snap" to data points
            self.hyperpol_span.extents = (
                hyperpol_times[start_idx] * 1000, 
                hyperpol_times[end_idx] * 1000
            )
            
            # Update the sliders in the action potential tab
            if hasattr(self.action_potential_tab, 'hyperpol_start'):
                self.action_potential_tab.hyperpol_start.set(start_idx)
            if hasattr(self.action_potential_tab, 'hyperpol_end'):
                self.action_potential_tab.hyperpol_end.set(end_idx)
            
            # Trigger the interval change handler
            if hasattr(self.action_potential_tab, 'on_integration_interval_change'):
                self.action_potential_tab.on_integration_interval_change()
            
            app_logger.debug(f"Hyperpol range snapped to points: {start_idx}-{end_idx}")
            
        except Exception as e:
            app_logger.error(f"Error updating hyperpol range: {str(e)}")

    def load_data(self):
        """Load data from file with proper cleanup of old data."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return
                
            app_logger.info(f"Loading file: {filepath}")
            
            # Clear existing data
            self.data = None
            self.time_data = None
            self.filtered_data = None
            self.processed_data = None
            self.orange_curve = None
            self.orange_curve_times = None
            
            # Store current file path
            self.current_file = filepath
            
            atf_handler = ATFHandler(filepath)
            atf_handler.load_atf()
            
            self.time_data = atf_handler.get_column("Time")
            self.data = atf_handler.get_column("#1")
            self.filtered_data = self.data.copy()
            
            # Update view limits
            self.view_tab.update_limits(
                t_min=self.time_data[0],
                t_max=self.time_data[-1],
                v_min=np.min(self.data),
                v_max=np.max(self.data)
            )
            
            self.analysis_tab.update_data(self.data, self.time_data)
            
            # Extract voltage from filename and initialize processor if found
            voltage = ActionPotentialProcessor.parse_voltage_from_filename(filepath)
            if voltage is not None:
                app_logger.info(f"Detected V2 voltage from filename: {voltage} mV")
                self.action_potential_tab.V2.set(voltage)  # Update UI value but don't run analysis
            else:
                self.action_potential_processor = None
            
            self.update_plot()
            filename = os.path.basename(filepath)
            self.status_var.set(f"Loaded: {filename}")
            
            # Check if this file exists in history
            if hasattr(self, 'history_manager') and hasattr(self.history_manager, 'history_entries'):
                history = self.history_manager.history_entries
                file_history = [entry for entry in history if entry['filename'] == filename]
                
                if file_history:
                    # Show most recent analysis results for this file
                    latest_result = file_history[-1]
                    msg = (
                        f"Previous analysis results for {filename}:\n"
                        f"Integral Value: {latest_result['integral_value']}\n"
                        f"Hyperpol Area: {latest_result['hyperpol_area']}\n"
                        f"Depol Area: {latest_result['depol_area']}\n"
                        f"Linear Capacitance: {latest_result['capacitance_nF']}"
                    )
                    messagebox.showinfo("File History", msg)
            
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
            
            self.filtered_data = combined_filter(self.data, **filters)
            
            self.update_plot()
            self.analysis_tab.update_filtered_data(self.filtered_data)
            
            self.window_manager.set_data(self.time_data, self.filtered_data)
            
            self.action_potential_processor = None
            
        except Exception as e:
            app_logger.error(f"Error applying filters: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply filters: {str(e)}")

    def on_analysis_update(self, analysis_params):
        """Handle changes in analysis settings"""
        if self.filtered_data is None:
            return
        try:
            self.analysis_tab.analyze_data(
                self.filtered_data, 
                self.time_data,
                analysis_params
            )
        except Exception as e:
            app_logger.error(f"Error updating analysis: {str(e)}")

    def on_display_mode_change(self, *args):
        """Handle changes in display mode to update span selectors."""
        try:
            # Check if we have active span selectors
            if not (hasattr(self, 'hyperpol_span') and hasattr(self, 'depol_span')):
                return
                
            # Check if points are enabled
            if not hasattr(self.action_potential_tab, 'show_points'):
                return
                
            show_points = self.action_potential_tab.show_points.get()
            if not show_points:
                return
                
            # Get current display mode
            display_mode = self.action_potential_tab.modified_display_mode.get()
            
            # Enable span selectors in both line and all_points modes
            enable_spans = display_mode in ["line", "all_points"]
            
            # Update span selector visibility
            self.hyperpol_span.set_visible(enable_spans)
            self.depol_span.set_visible(enable_spans)
            
            # Update canvas
            self.canvas.draw_idle()
            
            app_logger.debug(f"Span selectors updated for display mode {display_mode}")
            
        except Exception as e:
            app_logger.error(f"Error updating span selectors for display mode: {str(e)}")

    def on_action_potential_analysis(self, params):
        """
        Handle action potential analysis or display updates from the ActionPotentialTab.
        """
        # Handle visibility-only updates
        if isinstance(params, dict) and params.get('visibility_update', False):
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                self.update_plot_with_processed_data(
                    self.processed_data,
                    self.orange_curve,
                    self.orange_curve_times,
                    self.normalized_curve,
                    self.normalized_curve_times,
                    getattr(self, 'average_curve', None),
                    getattr(self, 'average_curve_times', None)
                )
            return

        # Validate data availability
        if self.filtered_data is None:
            messagebox.showwarning("Analysis", "No filtered data available")
            return

        try:
            app_logger.debug(f"Starting action potential analysis with params: {params}")
            
            # Initialize the processor
            self.action_potential_processor = ActionPotentialProcessor(
                self.filtered_data,
                self.time_data,
                params
            )

            # Run the main processing pipeline
            (
                processed_data,
                orange_curve,
                orange_times,
                normalized_curve,
                normalized_times,
                average_curve,
                average_curve_times,
                results
            ) = self.action_potential_processor.process_signal(
                use_alternative_method=params.get('use_alternative_method', False)
            )

            # Check for pipeline failure
            if processed_data is None:
                error_msg = results.get('error', 'Unknown error')
                messagebox.showwarning("Analysis", f"Analysis failed: {error_msg}")
                return

            # Store processed data for plotting
            self.processed_data = processed_data
            self.orange_curve = orange_curve
            self.orange_curve_times = orange_times
            self.normalized_curve = normalized_curve
            self.normalized_curve_times = normalized_times
            self.average_curve = average_curve
            self.average_curve_times = average_curve_times

            # Generate modified peaks (purple curves)
            (
                modified_hyperpol,
                modified_hyperpol_times,
                modified_depol,
                modified_depol_times
            ) = self.action_potential_processor.apply_average_to_peaks()

            # Store purple curve slice information from logs
            self.action_potential_processor._hyperpol_slice = (1035, 1235)
            self.action_potential_processor._depol_slice = (835, 1035)
            
            # Verify purple curves were created
            if (modified_hyperpol is None or modified_depol is None or 
                modified_hyperpol_times is None or modified_depol_times is None):
                app_logger.error("Failed to generate purple curves")
                messagebox.showwarning("Analysis", "Failed to generate purple curves")
                return

            # Calculate and integrate purple curves
            purple_results = self.action_potential_processor.calculate_purple_integrals()
            if isinstance(purple_results, dict):
                results.update(purple_results)

            # Update the plot with all curves
            self.update_plot_with_processed_data(
                processed_data,
                orange_curve,
                orange_times,
                normalized_curve,
                normalized_times,
                average_curve,
                average_curve_times
            )

            # Update results in the UI
            self.action_potential_tab.update_results(results)
            
            # Make sure the action_potential_processor is fully stored
            app_logger.debug(f"Analysis complete - action_potential_processor reference is {self.action_potential_processor is not None}")
            
            # PASS THE PROCESSOR REFERENCE DIRECTLY to avoid lookup issues
            if hasattr(self.action_potential_tab, 'set_processor'):
                self.action_potential_tab.set_processor(self.action_potential_processor)
            
            app_logger.info("Action potential analysis completed successfully")

        except Exception as e:
            app_logger.error(f"Error in action potential analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            if hasattr(self.action_potential_tab, 'disable_points_ui'):
                self.action_potential_tab.disable_points_ui()

    def plot_action_potential(self, processed_data, time_data):
        """Plot action potential analysis results"""
        try:
            self.ax.clear()
            self.ax.plot(self.time_data, self.data, 'b-', alpha=0.3, label='Original')
            if self.filtered_data is not None:
                self.ax.plot(self.time_data, self.filtered_data, 'r-', 
                             alpha=0.8, label='Filtered', linewidth=1.5)
            self.ax.plot(time_data, processed_data, 'g-', linewidth=1.5, label='Processed')
            self.ax.grid(True, which='both', linestyle='-', alpha=0.2)
            self.ax.set_ylabel('Current (pA)')
            self.ax.legend()
            self.format_time_axis()
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

    def format_time_axis(self):
        """Format time axis to always show milliseconds"""
        xlim = self.ax.get_xlim()

        self.ax.xaxis.set_major_formatter(lambda x, p: f"{x*1000:.0f}")
        self.ax.set_xlabel('Time (s)')
        time_range = xlim[1] - xlim[0]
        if time_range > 10:
            self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    def show_file_history_dialog(self, filename):
        """Show a dialog with previous analysis results for a file."""
        if not hasattr(self, 'history_manager') or not hasattr(self.history_manager, 'history_entries'):
            return
        
        history = self.history_manager.history_entries
        file_history = [entry for entry in history if entry['filename'] == filename]
        
        if not file_history:
            return
        
        # Create dialog window
        dialog = tk.Toplevel(self.master)
        dialog.title(f"File History: {filename}")
        dialog.transient(self.master)
        dialog.grab_set()
        dialog.geometry("400x300")
        
        # Create content frame
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill='both', expand=True)
        
        # Add icon
        icon_frame = ttk.Frame(frame)
        icon_frame.pack(fill='x', pady=10)
        
        info_icon = ttk.Label(icon_frame, text="ℹ", font=("Arial", 24))
        info_icon.pack(side='left', padx=10)
        
        # Add header
        header = ttk.Label(
            icon_frame, 
            text=f"Previous analysis results for {filename}:",
            font=("Arial", 12, "bold"),
            wraplength=300
        )
        header.pack(side='left', fill='x', expand=True)
        
        # Sort entries by timestamp (newest first)
        file_history.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_result = file_history[0]
        
        # Create content
        content_frame = ttk.Frame(frame)
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Add results
        results = [
            ("Integral Value:", latest_result['integral_value']),
            ("Hyperpol Area:", latest_result['hyperpol_area']),
            ("Depol Area:", latest_result['depol_area']),
            ("Linear Capacitance:", latest_result['capacitance_nF']),
            ("V2 Voltage:", latest_result.get('v2_voltage', "N/A")),
            ("Analysis Date:", latest_result['timestamp'])
        ]
        
        for i, (label, value) in enumerate(results):
            ttk.Label(content_frame, text=label, font=("Arial", 10, "bold")).grid(
                row=i, column=0, sticky='w', padx=5, pady=3
            )
            ttk.Label(content_frame, text=value, font=("Arial", 10)).grid(
                row=i, column=1, sticky='w', padx=5, pady=3
            )
        
        # Add button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=10)
        
        # Add progress bar to show up-to-date status
        progress_frame = ttk.Frame(frame)
        progress_frame.pack(fill='x', pady=5)
        
        progress = ttk.Progressbar(progress_frame, length=300, mode='determinate', value=100)
        progress.pack(side='top', fill='x', padx=10)
        
        status_text = "Analysis complete"
        status = ttk.Label(progress_frame, text=status_text, anchor='center')
        status.pack(side='top', fill='x', padx=10)
        
        # Close button
        ttk.Button(
            button_frame, 
            text="OK", 
            command=dialog.destroy,
            width=10
        ).pack(side='right', padx=10)
        
        # View History button
        ttk.Button(
            button_frame,
            text="View All History",
            command=lambda: [dialog.destroy(), self.show_analysis_history()]
        ).pack(side='left', padx=10)
        
        # Center dialog on parent window
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = self.master.winfo_rootx() + (self.master.winfo_width() - width) // 2
        y = self.master.winfo_rooty() + (self.master.winfo_height() - height) // 2
        dialog.geometry(f"+{x}+{y}")

    def update_plot(self, view_params=None):
        """Update plot with current data ensuring arrays match."""
        if self.data is None:
            return
        try:
            self.ax.clear()
            if view_params is None:
                view_params = self.view_tab.get_view_params()
            
            if view_params.get('use_interval', False):
                start_idx = np.searchsorted(self.time_data, view_params['t_min'])
                end_idx = np.searchsorted(self.time_data, view_params['t_max'])
                plot_time = self.time_data[start_idx:end_idx]
                plot_data = self.data[start_idx:end_idx]
                if self.filtered_data is not None:
                    plot_filtered = self.filtered_data[start_idx:end_idx]
                if hasattr(self, 'processed_data') and self.processed_data is not None:
                    plot_processed = self.processed_data[start_idx:end_idx]
            else:
                plot_time = self.time_data
                plot_data = self.data
                plot_filtered = self.filtered_data
                plot_processed = getattr(self, 'processed_data', None)

            if view_params.get('show_original', True):
                self.ax.plot(plot_time, plot_data, 'b-', label='Original Signal', alpha=0.3)
            if view_params.get('show_filtered', True) and plot_filtered is not None:
                if len(plot_filtered) == len(plot_time):
                    self.ax.plot(plot_time, plot_filtered, 'r-', label='Filtered Signal', alpha=0.5)
            if (plot_processed is not None and 
                hasattr(self.action_potential_tab, 'show_processed') and 
                self.action_potential_tab.show_processed.get()):
                display_mode = self.action_potential_tab.processed_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(plot_time, plot_processed, 'g-',
                                 label='Processed Signal' if display_mode=="line" else "_nolegend_",
                                 linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(plot_time, plot_processed, color='green', s=15, alpha=0.8, marker='.',
                                    label='Processed Signal' if display_mode=="points" else "_nolegend_")
            
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            if 'y_min' in view_params and 'y_max' in view_params:
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            if 't_min' in view_params and 't_max' in view_params:
                self.ax.set_xlim(view_params['t_min'], view_params['t_max'])
            
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            app_logger.error(f"Error updating plot: {str(e)}")
            raise

    def export_data(self):
        """Export the current data to a CSV file with detailed sections and integral values"""
        if self.filtered_data is None:
            messagebox.showwarning("Export", "No filtered data to export")
            return
            
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filepath:
                # Get original input filename
                input_filename = "No input file"
                if hasattr(self, 'current_file'):
                    input_filename = os.path.basename(self.current_file)
                
                # Get processor and results
                processor = getattr(self, 'action_potential_processor', None)
                if processor is None:
                    messagebox.showwarning("Export", "No processed data available")
                    return
                    
                # Get integral values
                results_dict = {
                    'integral_value': getattr(processor, 'integral_value', 'N/A'),
                    'hyperpol_area': getattr(processor, 'hyperpol_area', 'N/A'),
                    'depol_area': getattr(processor, 'depol_area', 'N/A'),
                    'purple_integral_value': getattr(processor, 'purple_integral_value', 'N/A')
                }
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    
                    # Write file info and V2
                    v2 = processor.params.get('V2', 0) if hasattr(processor, 'params') else 0
                    writer.writerow(["Input File:", input_filename, f"{v2}mV"])
                    writer.writerow([])
                    
                    # Write overall integral
                    writer.writerow(["Overall Integral (pC):", results_dict['integral_value']])
                    writer.writerow([])
                    
                    # Purple Hyperpol Section
                    writer.writerow(["PURPLE HYPERPOL CURVE"])
                    writer.writerow(["Hyperpol Integral:", results_dict['hyperpol_area']])
                    writer.writerow(["Index", "Hyperpol_pA", "Hyperpol_time_ms"])
                    
                    if hasattr(processor, 'modified_hyperpol') and hasattr(processor, 'modified_hyperpol_times'):
                        hyperpol = processor.modified_hyperpol
                        hyperpol_times = processor.modified_hyperpol_times
                        for i in range(len(hyperpol)):
                            writer.writerow([
                                i + 1,
                                f"{hyperpol[i]:.7f}",
                                f"{hyperpol_times[i]*1000:.7f}"
                            ])
                    writer.writerow([])
                    
                    # Purple Depol Section
                    writer.writerow(["PURPLE DEPOL CURVE"])
                    writer.writerow(["Depol Integral:", results_dict['depol_area']])
                    writer.writerow(["Index", "Depol_pA", "Depol_time_ms"])
                    
                    if hasattr(processor, 'modified_depol') and hasattr(processor, 'modified_depol_times'):
                        depol = processor.modified_depol
                        depol_times = processor.modified_depol_times
                        for i in range(len(depol)):
                            writer.writerow([
                                i + 1,
                                f"{depol[i]:.7f}",
                                f"{depol_times[i]*1000:.7f}"
                            ])
                    writer.writerow([])
                    
                    # Purple Integral Summary
                    writer.writerow(["Purple Integral Summary:", results_dict['purple_integral_value']])
                    
                    app_logger.info(f"Data exported to {filepath}")
                    messagebox.showinfo("Export", "Data exported successfully")
                    
        except Exception as e:
            app_logger.error(f"Error exporting data: {str(e)}")
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def export_figure(self):
        """Export the current figure"""
        try:
            filepath = filedialog.asksaveasfilename(defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")])
            if filepath:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                app_logger.info(f"Figure exported to {filepath}")
                messagebox.showinfo("Export", "Figure exported successfully")
        except Exception as e:
            app_logger.error(f"Error exporting figure: {str(e)}")
            messagebox.showerror("Error", f"Failed to export figure: {str(e)}")

    def update_plot_with_processed_data(
        self,
        processed_data,
        orange_curve,
        orange_times,
        normalized_curve,
        normalized_times,
        average_curve,
        average_curve_times
    ):
        """Update the main plot with all processed data and curves."""
        try:
            if not hasattr(self, 'action_potential_tab'):
                return

            self.ax.clear()
            view_params = self.view_tab.get_view_params()
            display_options = self.action_potential_tab.get_parameters().get('display_options', {})

            # Get integration ranges and points visibility
            intervals = self.action_potential_tab.get_intervals()
            show_points = intervals.get('show_points', False)
            integration_ranges = intervals.get('integration_ranges', {})
            
            app_logger.debug(f"Updating plot with integration ranges: {integration_ranges}")
            app_logger.debug(f"Points visibility: {show_points}")

            # Original signal
            if display_options.get('show_noisy_original', False):
                self.ax.plot(self.time_data * 1000, self.data, 'b-', 
                        label='Original Signal', alpha=0.3)

            # Filtered signal in maroon
            if self.filtered_data is not None and display_options.get('show_red_curve', True):
                self.ax.plot(self.time_data * 1000, self.filtered_data, 
                        color='#800000',  # Maroon color
                        label='Filtered Signal', 
                        alpha=0.7,
                        linewidth=1.5)

            # Processed signal
            if processed_data is not None and display_options.get('show_processed', True):
                display_mode = self.action_potential_tab.processed_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(self.time_data * 1000, processed_data, 'g-', 
                            label='Processed Signal',
                            linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(self.time_data * 1000, processed_data, 
                                color='green', s=15, alpha=0.8, 
                                marker='.', label='Processed Points')

            # 50-point average (orange)
            if (orange_curve is not None and orange_times is not None
                and display_options.get('show_average', True)):
                display_mode = self.action_potential_tab.average_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(orange_times * 1000, orange_curve, 
                            color='#FFA500',  # Orange
                            label='50-point Average', 
                            linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(orange_times * 1000, orange_curve, 
                                color='#FFA500', s=25, alpha=1, 
                                marker='o', label='Average Points')

            # Voltage-normalized (dark blue)
            if (normalized_curve is not None and normalized_times is not None
                and display_options.get('show_normalized', True)):
                display_mode = self.action_potential_tab.normalized_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(normalized_times * 1000, normalized_curve, 
                            color='#0057B8',  # Dark blue
                            label='Voltage-Normalized', 
                            linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(normalized_times * 1000, normalized_curve, 
                                color='#0057B8', s=25, alpha=1,
                                marker='o', label='Normalized Points')

            # Averaged-normalized (magenta)
            if (average_curve is not None and average_curve_times is not None
                and display_options.get('show_averaged_normalized', True)):
                display_mode = self.action_potential_tab.averaged_normalized_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(average_curve_times * 1000, average_curve, 
                            color='magenta',
                            label='Averaged Normalized', 
                            linewidth=2, alpha=0.8)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(average_curve_times * 1000, average_curve, 
                                color='magenta', s=30, alpha=1,
                                marker='o', label='Avg Normalized Points')

            # Modified peaks (purple) with integration ranges
            has_purple_curves = (hasattr(self.action_potential_processor, 'modified_hyperpol') and 
                            hasattr(self.action_potential_processor, 'modified_depol') and
                            self.action_potential_processor.modified_hyperpol is not None and 
                            self.action_potential_processor.modified_depol is not None)

            if has_purple_curves and display_options.get('show_modified', True):
                display_mode = self.action_potential_tab.modified_display_mode.get()
                
                # Get hyperpolarization data
                hyperpol = self.action_potential_processor.modified_hyperpol
                hyperpol_times = self.action_potential_processor.modified_hyperpol_times
                
                # Get depolarization data
                depol = self.action_potential_processor.modified_depol
                depol_times = self.action_potential_processor.modified_depol_times
                
                # Plot curves based on display mode
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(hyperpol_times * 1000, hyperpol,
                            color='purple', label='Modified Peaks', 
                            linewidth=2, alpha=0.8)
                    self.ax.plot(depol_times * 1000, depol,
                            color='purple', label='_nolegend_',
                            linewidth=2, alpha=0.8)
                            
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(hyperpol_times * 1000, hyperpol,
                                color='purple', s=30, alpha=0.8,
                                marker='o',
                                label='_nolegend_' if display_mode == "all_points" else "Modified Points")
                    self.ax.scatter(depol_times * 1000, depol,
                                color='purple', s=30, alpha=0.8,
                                marker='o',
                                label='_nolegend_')
                
                # Visualize integration ranges as shaded areas
                if integration_ranges:
                    # Hyperpolarization range
                    if 'hyperpol' in integration_ranges:
                        hyperpol_range = integration_ranges['hyperpol']
                        start_idx = hyperpol_range['start']
                        end_idx = hyperpol_range['end']
                        
                        if 0 <= start_idx < len(hyperpol_times) and 0 < end_idx <= len(hyperpol_times):
                            self.ax.axvspan(
                                hyperpol_times[start_idx] * 1000,
                                hyperpol_times[end_idx-1] * 1000,
                                color='blue', alpha=0.15,
                                label='Hyperpol Range'
                            )
                            app_logger.debug(f"Added hyperpol range visualization: {start_idx}-{end_idx}")
                    
                    # Depolarization range
                    if 'depol' in integration_ranges:
                        depol_range = integration_ranges['depol']
                        start_idx = depol_range['start']
                        end_idx = depol_range['end']
                        
                        if 0 <= start_idx < len(depol_times) and 0 < end_idx <= len(depol_times):
                            self.ax.axvspan(
                                depol_times[start_idx] * 1000,
                                depol_times[end_idx-1] * 1000,
                                color='red', alpha=0.15,
                                label='Depol Range'
                            )
                            app_logger.debug(f"Added depol range visualization: {start_idx}-{end_idx}")
                            
                # Add regression lines if enabled and regression intervals exist
                if show_points and hasattr(self, 'plot_regression_lines'):
                    regression_interval = intervals.get('regression_interval')
                    if regression_interval:
                        # Plot regression line for hyperpolarization
                        self.plot_regression_lines(
                            hyperpol, hyperpol_times, 
                            regression_interval, 
                            color='blue', alpha=0.8
                        )
                        
                        # Plot regression line for depolarization
                        self.plot_regression_lines(
                            depol, depol_times, 
                            regression_interval, 
                            color='red', alpha=0.8
                        )

            # Configure axes and layout
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()

            # Apply view limits if set
            if view_params.get('use_custom_ylim', False):
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            if view_params.get('use_interval', False):
                self.ax.set_xlim(view_params['t_min'] * 1000, view_params['t_max'] * 1000)

            self.fig.tight_layout()
            self.canvas.draw_idle()
            
            # Toggle span selector visibility based on display mode
            active_mode = False
            if hasattr(self.action_potential_tab, 'modified_display_mode'):
                # Enable in both line and all_points modes
                active_mode = (self.action_potential_tab.modified_display_mode.get() in ["line", "all_points"])

            # Enable spans if show_points is true and we're in a compatible display mode
            enable_spans = show_points and active_mode

            if hasattr(self, 'span_selector'):
                self.span_selector.set_visible(enable_spans)
                self.span_selector.set_active(enable_spans)
                app_logger.debug(f"Span selector visibility set to {enable_spans}")

            # If span selectors are visible and ranges are defined, position them
            if enable_spans and hasattr(self, 'hyperpol_span') and hasattr(self, 'depol_span'):
                # Make sure we have the times data
                if hasattr(self.action_potential_processor, 'modified_hyperpol_times') and \
                   hasattr(self.action_potential_processor, 'modified_depol_times'):
                    
                    hyperpol_times = self.action_potential_processor.modified_hyperpol_times
                    depol_times = self.action_potential_processor.modified_depol_times
                    
                    # Hyperpol range
                    if 'hyperpol' in integration_ranges:
                        hyperpol_range = integration_ranges['hyperpol']
                        start_idx = hyperpol_range['start']
                        end_idx = hyperpol_range['end']
                        
                        if (hyperpol_times is not None and 0 <= start_idx < len(hyperpol_times) and 
                            0 < end_idx <= len(hyperpol_times)):
                            # Set the initial position of the hyperpol span selector
                            try:
                                # Calculate extents based on actual data points
                                new_extents = (
                                    hyperpol_times[start_idx] * 1000, 
                                    hyperpol_times[end_idx - 1] * 1000
                                )
                                
                                # Only update if the extents have changed from sliders
                                # This prevents "jumpy" behavior when dragging
                                if not hasattr(self, 'prev_hyperpol_extents') or self.prev_hyperpol_extents != new_extents:
                                    self.hyperpol_span.extents = new_extents
                                    self.prev_hyperpol_extents = new_extents
                                    
                                self.hyperpol_span.set_visible(True)
                                self.hyperpol_span.set_active(True)
                                
                                app_logger.debug(f"Hyperpol span positioned at: {start_idx}-{end_idx}")
                            except Exception as e:
                                app_logger.error(f"Error setting hyperpol span extents: {str(e)}")
                    
                    # Depol range
                    if 'depol' in integration_ranges:
                        depol_range = integration_ranges['depol']
                        start_idx = depol_range['start']
                        end_idx = depol_range['end']
                        
                        if (depol_times is not None and 0 <= start_idx < len(depol_times) and 
                            0 < end_idx <= len(depol_times)):
                            # Set the initial position of the depol span selector
                            try:
                                # Calculate extents based on actual data points
                                new_extents = (
                                    depol_times[start_idx] * 1000, 
                                    depol_times[end_idx - 1] * 1000
                                )
                                
                                # Only update if the extents have changed from sliders
                                # This prevents "jumpy" behavior when dragging
                                if not hasattr(self, 'prev_depol_extents') or self.prev_depol_extents != new_extents:
                                    self.depol_span.extents = new_extents
                                    self.prev_depol_extents = new_extents
                                    
                                self.depol_span.set_visible(True)
                                self.depol_span.set_active(True)
                                
                                app_logger.debug(f"Depol span positioned at: {start_idx}-{end_idx}")
                            except Exception as e:
                                app_logger.error(f"Error setting depol span extents: {str(e)}")
            else:
                # Disable span selectors when not in the right mode
                if hasattr(self, 'hyperpol_span'):
                    self.hyperpol_span.set_visible(False)
                    self.hyperpol_span.set_active(False)
                if hasattr(self, 'depol_span'):
                    self.depol_span.set_visible(False)
                    self.depol_span.set_active(False)
                
            app_logger.debug("Plot updated with all processed data and integration ranges")

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
