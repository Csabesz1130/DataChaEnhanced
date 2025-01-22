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
        """
        Find closest orange point using both time and value differences.
        
        Args:
            x_ms: Cursor x position in milliseconds
            y_pA: Cursor y position in pA
            
        Returns:
            tuple: (point_number, time_ms) or (None, None)
        """
        if not hasattr(self, 'orange_curve_times') or self.orange_curve_times is None:
            return None, None

        # Convert cursor time to seconds for comparison with stored times
        x_sec = x_ms / 100.0  # Convert from ms to seconds (*100 plot scale)
        
        # Find time differences
        time_diffs = np.abs(self.orange_curve_times - x_sec)
        TIME_THRESHOLD = 2.0 / 100.0  # 2ms in seconds, accounting for *100 scale
        
        # Get points within time threshold
        close_mask = time_diffs < TIME_THRESHOLD
        if not np.any(close_mask):
            return None, None
            
        # Among time-close points, find closest by value
        close_indices = np.where(close_mask)[0]
        value_diffs = np.abs(self.orange_curve[close_indices] - y_pA)
        best_idx = close_indices[np.argmin(value_diffs)]
        
        # Get point info (convert back to ms for display)
        point_time_ms = self.orange_curve_times[best_idx] * 100.0  # *100 for plot scale
        point_number = best_idx + 1  # Convert to 1-based indexing
        
        app_logger.debug(
            f"Found orange point {point_number} at {point_time_ms:.1f}ms "
            f"(cursor: {x_ms:.1f}ms, {y_pA:.1f}pA)"
        )
        
        return point_number, point_time_ms

    def on_mouse_move(self, event):
        """Handle mouse movement with accurate point identification."""
        if event.inaxes != self.ax:
            self.cursor_label.config(text="")
            return

        if self.time_data is None:
            return

        # Get cursor position
        x_ms = event.xdata  # Already in milliseconds
        y_pA = event.ydata

        # Build cursor text
        cursor_text = f"Time: {x_ms:.1f} ms, Current: {y_pA:.1f} pA"

        # Find closest orange point
        result = self._find_closest_point(x_ms, y_pA)
        if result[0] is not None:  # If a point was found
            point_num, _ = result
            cursor_text += f", Orange Point: {point_num}"

        # Update display
        self.cursor_label.config(text=cursor_text)

    def load_data(self):
        """Load data from file with proper cleanup of old data."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return
                
            app_logger.info(f"Loading file: {filepath}")
            
            self.data = None
            self.time_data = None
            self.filtered_data = None
            self.processed_data = None
            self.orange_curve = None
            self.orange_curve_times = None
            
            atf_handler = ATFHandler(filepath)
            atf_handler.load_atf()
            
            self.time_data = atf_handler.get_column("Time")
            self.data = atf_handler.get_column("#1")
            self.filtered_data = self.data.copy()
            
            self.view_tab.update_limits(
                t_min=self.time_data[0],
                t_max=self.time_data[-1],
                v_min=np.min(self.data),
                v_max=np.max(self.data)
            )
            
            self.analysis_tab.update_data(self.data, self.time_data)
            
            self.action_potential_processor = None
            
            self.update_plot()
            filename = os.path.basename(filepath)
            self.status_var.set(f"Loaded: {filename}")
            
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

    def on_action_potential_analysis(self, params):
        """
        Handle action potential analysis or display updates from the ActionPotentialTab.
        """
        # 1) If this is only a "visibility" request, just re-draw using existing data
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

        # 2) If no filtered data, cannot analyze
        if self.filtered_data is None:
            messagebox.showwarning("Analysis", "No filtered data available")
            return

        try:
            # 3) Create the ActionPotentialProcessor with current data & parameters
            self.action_potential_processor = ActionPotentialProcessor(
                self.filtered_data,
                self.time_data,
                params
            )

            # 4) Run main pipeline (baseline → normalization → ... → integration)
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

            # 5) If the pipeline itself failed
            if processed_data is None:
                messagebox.showwarning(
                    "Analysis",
                    "Analysis failed: " + str(results.get('integral_value', 'Unknown error'))
                )
                return

            # 6) Store all these arrays in self for re-plotting
            self.processed_data = processed_data
            self.orange_curve = orange_curve
            self.orange_curve_times = orange_times
            self.normalized_curve = normalized_curve
            self.normalized_curve_times = normalized_times
            self.average_curve = average_curve
            self.average_curve_times = average_curve_times

            # 7) Now produce the "Modified Peaks" (the purple curves)
            #    The call returns 4 arrays; store them on the processor so update_plot_with_processed_data can see them.
            (
                modified_hyperpol,
                modified_hyperpol_times,
                modified_depol,
                modified_depol_times
            ) = self.action_potential_processor.apply_average_to_peaks()

            # If everything worked, the processor now has .modified_hyperpol, etc. attached
            # No need to do anything else as your update_plot_with_processed_data() checks self.action_potential_processor

            # 8) Update the plot with *all* relevant arrays
            self.update_plot_with_processed_data(
                processed_data,
                orange_curve,
                orange_times,
                normalized_curve,
                normalized_times,
                average_curve,
                average_curve_times
            )

            # 9) Show results in the ActionPotentialTab (integral, etc.)
            self.action_potential_tab.update_results(results)

            app_logger.info("Action potential analysis completed successfully")

        except Exception as e:
            app_logger.error(f"Error in action potential analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

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
        self.ax.set_xlabel('Time (ms)')
        time_range = xlim[1] - xlim[0]
        if time_range > 10:
            self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))

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
                self.ax.plot(plot_time * 1000, plot_data, 'b-', label='Original Signal', alpha=0.3)
            if view_params.get('show_filtered', True) and plot_filtered is not None:
                if len(plot_filtered) == len(plot_time):
                    self.ax.plot(plot_time * 1000, plot_filtered, 'r-', label='Filtered Signal', alpha=0.5)
            if (plot_processed is not None and 
                hasattr(self.action_potential_tab, 'show_processed') and 
                self.action_potential_tab.show_processed.get()):
                display_mode = self.action_potential_tab.processed_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(plot_time * 1000, plot_processed, 'g-',
                                 label='Processed Signal' if display_mode=="line" else "_nolegend_",
                                 linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(plot_time * 1000, plot_processed, color='green', s=15, alpha=0.8, marker='.',
                                    label='Processed Signal' if display_mode=="points" else "_nolegend_")
            
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            if 'y_min' in view_params and 'y_max' in view_params:
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            if 't_min' in view_params and 't_max' in view_params:
                self.ax.set_xlim(view_params['t_min'] * 1000, view_params['t_max'] * 1000)
            
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
            filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if filepath:
                df = pd.DataFrame({
                    'Time': self.time_data,
                    'Original': self.data,
                    'Filtered': self.filtered_data
                })
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
        """
        Update the main plot with processed data, orange curve, normalized curve,
        average curve, and any other specialized curves (modified peaks).
        """
        try:
            if not hasattr(self, 'action_potential_tab'):
                return

            self.ax.clear()
            view_params = self.view_tab.get_view_params()
            display_options = self.action_potential_tab.get_parameters().get('display_options', {})

            # 1) Original signal if requested
            if display_options.get('show_noisy_original', False):
                self.ax.plot(self.time_data * 100, self.data, 'b-', label='Original Signal', alpha=0.3)

            # 2) Filtered signal
            if self.filtered_data is not None:
                self.ax.plot(self.time_data * 100, self.filtered_data, 'r-', label='Filtered Signal', alpha=0.5)

            # 3) Processed
            if processed_data is not None and display_options.get('show_processed', True):
                display_mode = self.action_potential_tab.processed_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(self.time_data * 100, processed_data, 'g-', label='Processed Signal',
                                 linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(self.time_data * 100, processed_data, color='green', s=15, alpha=0.8, marker='.',
                                    label='Processed Points')

            # 4) 50-point average (orange)
            if (orange_curve is not None and orange_times is not None
                and display_options.get('show_average', True)):
                display_mode = self.action_potential_tab.average_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(orange_times * 100, orange_curve, color='orange',
                                 label='50-point Average', linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(orange_times * 100, orange_curve, color='orange', s=25, alpha=1, marker='o',
                                    label='Average Points')

            # 5) Voltage-normalized (dark blue)
            if (normalized_curve is not None and normalized_times is not None
                and display_options.get('show_normalized', True)):
                display_mode = self.action_potential_tab.normalized_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(normalized_times * 100, normalized_curve, color='darkblue',
                                 label='Voltage-Normalized', linewidth=1.5, alpha=0.7)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(normalized_times * 100, normalized_curve, color='darkblue', s=25, alpha=1,
                                    marker='o', label='Normalized Points')

            # 6) Averaged-normalized (magenta)
            if (average_curve is not None and average_curve_times is not None
                and display_options.get('show_averaged_normalized', True)):
                display_mode = self.action_potential_tab.averaged_normalized_display_mode.get()
                if display_mode in ["line", "all_points"]:
                    self.ax.plot(average_curve_times * 100, average_curve, color='magenta',
                                 label='Averaged Normalized', linewidth=2, alpha=0.8)
                if display_mode in ["points", "all_points"]:
                    self.ax.scatter(average_curve_times * 100, average_curve, color='magenta', s=30, alpha=1,
                                    marker='o', label='Avg Normalized Points')

            # 7) Plot "modified" peaks if applicable
            if display_options.get('show_modified', True):
                display_mode = self.action_potential_tab.modified_display_mode.get()
                # hyperpol
                if (hasattr(self.action_potential_processor, 'modified_hyperpol')
                    and self.action_potential_processor.modified_hyperpol is not None):
                    if display_mode in ["line", "all_points"]:
                        self.ax.plot(self.action_potential_processor.modified_hyperpol_times * 100,
                                     self.action_potential_processor.modified_hyperpol,
                                     color='purple', label='Modified Peaks', linewidth=2, alpha=0.8)
                    if display_mode in ["points", "all_points"]:
                        self.ax.scatter(self.action_potential_processor.modified_hyperpol_times * 100,
                                        self.action_potential_processor.modified_hyperpol,
                                        color='purple', s=30, alpha=0.8)
                # depol
                if (hasattr(self.action_potential_processor, 'modified_depol')
                    and self.action_potential_processor.modified_depol is not None):
                    if display_mode in ["line", "all_points"]:
                        self.ax.plot(self.action_potential_processor.modified_depol_times * 100,
                                     self.action_potential_processor.modified_depol,
                                     color='purple', label='_nolegend_',
                                     linewidth=2, alpha=0.8)
                    if display_mode in ["points", "all_points"]:
                        self.ax.scatter(self.action_potential_processor.modified_depol_times * 100,
                                        self.action_potential_processor.modified_depol,
                                        color='purple', s=30, alpha=0.8)

            # Final labeling
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()

            if view_params.get('use_custom_ylim', False):
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
            if view_params.get('use_interval', False):
                self.ax.set_xlim(view_params['t_min'] * 100, view_params['t_max'] * 100)

            self.fig.tight_layout()
            self.canvas.draw_idle()
            app_logger.debug("Plot updated with all processed data")

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
