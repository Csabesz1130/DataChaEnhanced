import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import gc
import weakref
import matplotlib

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
from src.excel_charted.dual_curves_export_integration import add_dual_excel_export_to_app
from src.csv_export.dual_curves_csv_export import add_csv_export_buttons
from src.gui.direct_spike_removal import remove_spikes_from_processor
from src.gui.simplified_set_exporter import add_set_export_to_toolbar
from src.gui.batch_set_exporter import add_set_export_to_toolbar
from src.gui.multi_file_analysis import add_multi_file_analysis_to_toolbar
from src.gui.curve_fitting_gui import CurveFittingPanel
from src.utils.hot_reload import initialize_hot_reload, stop_hot_reload

class SignalAnalyzerApp:
    def __init__(self, master):
        """Initialize the Signal Analyzer application with memory optimization."""
        self.master = master
        self.master.title("Signal Analyzer - Memory Optimized")
        
        # Initialize data variables with memory management
        self._data = None
        self._time_data = None
        self._filtered_data = None
        self._processed_data = None
        self._orange_curve = None
        self._orange_curve_times = None
        
        # Memory management
        self._memory_usage = 0
        self.active_figures = weakref.WeakSet()
        
        self.current_filters = {}
        self.action_potential_processor = None
        self.current_file = None

        self.use_regression_hyperpol = tk.BooleanVar(value=False)
        self.use_regression_depol = tk.BooleanVar(value=False)

        self.saved_plot_limits = {
        'xlim': None,
        'ylim': None,
        'auto_restore': True  # Flag to enable/disable auto-restore
        }
        
        # Create main layout
        self.setup_main_layout()
        
        # Initialize window manager BEFORE toolbar setup
        self.window_manager = SignalWindowManager(self.master)
        
        # Initialize history manager
        self.history_manager = AnalysisHistoryManager(self)
        
        # Setup components
        self.setup_menubar()
        self.setup_toolbar()
        self.setup_plot()
        self.setup_plot_interaction()
        self.setup_tabs()

        # Setup hot reload for development
        self.setup_hot_reload()

        # Setup memory management
        self.setup_memory_management()

        add_excel_export_to_app(self)
        
        # Add the dual curves export functionality
        add_dual_excel_export_to_app(self)
        
        # Add CSV export functionality
        add_csv_export_buttons(self)

        # Fix window sizing issues
        self.fix_window_sizing()
    
        # Initialize curve fitting (after plot and tabs are created)
        self.master.after(500, self.initialize_curve_fitting)
        
        app_logger.info("Application initialized successfully with memory optimization")

    # Property-based data management with automatic cleanup
    @property
    def data(self):
        """Property getter for data"""
        return self._data
    
    @data.setter 
    def data(self, value):
        """Property setter for data with automatic cleanup"""
        # Clean up old data
        if self._data is not None:
            old_size = self._data.nbytes if hasattr(self._data, 'nbytes') else 0
            app_logger.debug(f"Cleaning up old data array: {old_size / 1024 / 1024:.1f} MB")
            del self._data
            self._memory_usage -= old_size
            
        # Store new data
        if value is not None:
            self._data = np.asarray(value, dtype=np.float64)
            # Ensure contiguous memory layout for efficiency
            if not self._data.flags['C_CONTIGUOUS']:
                old_data = self._data
                self._data = np.ascontiguousarray(self._data)
                del old_data
                
            new_size = self._data.nbytes
            self._memory_usage += new_size
            app_logger.debug(f"New data stored: {self._data.shape}, {new_size / 1024 / 1024:.1f} MB")
        else:
            self._data = None
            
        # Force garbage collection after data change
        gc.collect()

    @property
    def time_data(self):
        """Property getter for time data"""
        return self._time_data
    
    @time_data.setter
    def time_data(self, value):
        """Property setter for time data with cleanup"""
        if self._time_data is not None:
            del self._time_data
            
        if value is not None:
            self._time_data = np.asarray(value, dtype=np.float64)
        else:
            self._time_data = None
            
        gc.collect()
    
    @property 
    def filtered_data(self):
        """Property getter for filtered data"""
        return self._filtered_data
    
    @filtered_data.setter
    def filtered_data(self, value):
        """Property setter for filtered data with cleanup"""
        if self._filtered_data is not None:
            del self._filtered_data
            
        if value is not None:
            self._filtered_data = np.asarray(value, dtype=np.float64)
        else:
            self._filtered_data = None
            
        gc.collect()

    @property
    def processed_data(self):
        """Property getter for processed data"""
        return self._processed_data
    
    @processed_data.setter
    def processed_data(self, value):
        """Property setter for processed data with cleanup"""
        if self._processed_data is not None:
            del self._processed_data
            
        if value is not None:
            self._processed_data = np.asarray(value, dtype=np.float64)
        else:
            self._processed_data = None
            
        gc.collect()

    @property
    def orange_curve(self):
        """Property getter for orange curve"""
        return self._orange_curve
    
    @orange_curve.setter
    def orange_curve(self, value):
        """Property setter for orange curve with cleanup"""
        if self._orange_curve is not None:
            del self._orange_curve
            
        if value is not None:
            self._orange_curve = np.asarray(value, dtype=np.float64)
        else:
            self._orange_curve = None
            
        gc.collect()

    @property
    def orange_curve_times(self):
        """Property getter for orange curve times"""
        return self._orange_curve_times
    
    @orange_curve_times.setter
    def orange_curve_times(self, value):
        """Property setter for orange curve times with cleanup"""
        if self._orange_curve_times is not None:
            del self._orange_curve_times
            
        if value is not None:
            self._orange_curve_times = np.asarray(value, dtype=np.float64)
        else:
            self._orange_curve_times = None
            
        gc.collect()

    def setup_memory_management(self):
        """Setup memory management and monitoring"""
        # Configure matplotlib for memory efficiency
        matplotlib.rcParams['figure.max_open_warning'] = 5
        
        # Set up periodic cleanup
        self.master.after(30000, self.periodic_cleanup)  # Every 30 seconds
        
        # Bind cleanup to window close
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        app_logger.debug("Memory management setup completed")

    def periodic_cleanup(self):
        """Periodic memory cleanup function"""
        try:
            # Close any orphaned matplotlib figures
            self.cleanup_matplotlib_figures()
            
            # Force garbage collection
            collected = gc.collect()
            if collected > 0:
                app_logger.debug(f"Periodic cleanup: collected {collected} objects")
            
            # Check memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > 500:  # Warning threshold
                    app_logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
                    self.force_cleanup()
            except ImportError:
                pass
            
            # Schedule next cleanup
            self.master.after(30000, self.periodic_cleanup)
            
        except Exception as e:
            app_logger.error(f"Error in periodic cleanup: {str(e)}")
            # Still schedule next cleanup even if this one failed
            self.master.after(30000, self.periodic_cleanup)

    def initialize_curve_fitting(self):
        """Initialize curve fitting functionality after plot and tabs are created."""
        try:
            if (hasattr(self, 'fig') and hasattr(self, 'ax') and 
                hasattr(self, 'action_potential_tab')):
                
                # Create curve fitting panel in the action potential tab's scrollable frame
                parent_frame = self.action_potential_tab.scrollable_frame
                
                # Create the panel
                self.curve_fitting_panel = CurveFittingPanel(parent_frame, self)
                
                # Initialize with the main app's figure and axes
                self.curve_fitting_panel.initialize_fitting_manager(self.fig, self.ax)
                
                # Store reference in action potential tab too
                self.action_potential_tab.curve_fitting_panel = self.curve_fitting_panel
                
                app_logger.info("Curve fitting panel initialized successfully")
                
        except Exception as e:
            app_logger.error(f"Error initializing curve fitting: {str(e)}")

    def fix_window_sizing(self):
        """Fix window sizing to allow proper zooming and resizing."""
        try:
            # Remove size constraints and make properly resizable
            self.master.minsize(800, 600)
            self.master.resizable(True, True)
            
            # Set window to 80% of screen size
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)
            
            # Center window
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            self.master.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            # Add zoom controls
            def zoom_in(event=None):
                w, h = self.master.winfo_width(), self.master.winfo_height()
                new_w, new_h = int(w*1.1), int(h*1.1)
                # Limit to screen size
                new_w = min(new_w, screen_width - 100)
                new_h = min(new_h, screen_height - 100)
                self.master.geometry(f"{new_w}x{new_h}")
            
            def zoom_out(event=None):
                w, h = self.master.winfo_width(), self.master.winfo_height()
                new_w = max(800, int(w*0.9))
                new_h = max(600, int(h*0.9))
                self.master.geometry(f"{new_w}x{new_h}")
            
            def reset_zoom(event=None):
                self.master.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            def toggle_fullscreen(event=None):
                current = self.master.attributes('-fullscreen')
                self.master.attributes('-fullscreen', not current)
            
            # Bind keyboard shortcuts
            self.master.bind('<Control-plus>', zoom_in)
            self.master.bind('<Control-equal>', zoom_in)  # For keyboards without numpad
            self.master.bind('<Control-minus>', zoom_out)
            self.master.bind('<Control-0>', reset_zoom)
            self.master.bind('<F11>', toggle_fullscreen)
            self.master.bind('<Escape>', lambda e: self.master.attributes('-fullscreen', False))
            
            app_logger.info("Window sizing fixed with zoom controls (Ctrl+/-, F11, Ctrl+0)")
            
        except Exception as e:
            app_logger.error(f"Error fixing window sizing: {str(e)}")

    def cleanup_matplotlib_figures(self):
        """Clean up matplotlib figures to prevent memory leaks"""
        try:
            # Get list of all figure numbers
            figure_numbers = plt.get_fignums()
            
            if len(figure_numbers) > 3:  # Keep max 3 figures
                app_logger.debug(f"Cleaning up {len(figure_numbers)} matplotlib figures")
                
                # Close excess figures (keep the 3 most recent)
                for fig_num in figure_numbers[:-3]:
                    plt.close(fig_num)
                
                # Force cleanup
                gc.collect()
                
        except Exception as e:
            app_logger.error(f"Error cleaning matplotlib figures: {str(e)}")

    def force_cleanup(self):
        """Force memory cleanup (for debugging and emergency situations)"""
        try:
            app_logger.info("Forcing memory cleanup")
            
            # Close matplotlib figures
            self.cleanup_matplotlib_figures()
            
            # Force garbage collection multiple times
            total_collected = 0
            for i in range(3):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            # Log memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                app_logger.info(f"Force cleanup: collected {total_collected} objects, current usage: {memory_mb:.1f} MB")
            except ImportError:
                app_logger.info(f"Force cleanup: collected {total_collected} objects")
                
        except Exception as e:
            app_logger.error(f"Error in force cleanup: {str(e)}")

    def clear_all_data(self):
        """Clear all data and free memory"""
        app_logger.debug("Clearing all data")
        
        # Clear data using properties (automatic cleanup)
        self.data = None
        self.time_data = None
        self.filtered_data = None
        self.processed_data = None
        self.orange_curve = None
        self.orange_curve_times = None
        
        # Clear other data structures
        if hasattr(self, 'normalized_curve'):
            self.normalized_curve = None
        if hasattr(self, 'average_curve'):
            self.average_curve = None
            
        # Clear processor
        self.action_potential_processor = None
        
        # Clear file reference
        self.current_file = None
        
        # Force garbage collection
        gc.collect()
        
        app_logger.debug("All data cleared successfully")

    def _configure_axes(self):
        """Configure both X and Y axes with proper labels and ticks"""
        try:
            # Set labels
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            
            # Ensure y-axis ticks are visible
            self.ax.tick_params(axis='y', which='major', labelsize=9, pad=5)
            self.ax.tick_params(axis='x', which='major', labelsize=9, pad=5)
            
            # Force y-axis tick locator and formatter
            from matplotlib.ticker import MaxNLocator, ScalarFormatter
            
            # Set y-axis locator to ensure ticks are shown
            self.ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
            self.ax.yaxis.set_major_formatter(ScalarFormatter())
            
            # Set x-axis locator
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
            
            # Ensure ticks are visible
            self.ax.yaxis.set_visible(True)
            self.ax.xaxis.set_visible(True)
            
            # Force redraw of axis
            self.ax.figure.canvas.draw_idle()
            
        except Exception as e:
            app_logger.error(f"Error configuring axes: {str(e)}")

    def setup_menubar(self):
        """Setup the application menubar"""
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)
        
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_command(label="Export Figure", command=self.export_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="View History", command=self.show_analysis_history)
        analysis_menu.add_command(label="Export Purple Curves", command=self.on_export_purple_curves)
        
        # Help menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Check for Updates", command=self.check_for_updates)

    def show_about(self):
        """Show about dialog with AI status"""
        messagebox.showinfo(
            "About Signal Analyzer",
            f"Signal Analyzer v1.0\n\n"
            f"Advanced electrophysiology data analysis tool\n\n"
            f"Build: 2024.1"
        )

    # AI Integration Methods (only if AI is available)
    def run_ai_analysis(self):
        """Switch to AI tab and run AI analysis"""
        if hasattr(self, 'tabs') and 'ai_analysis' in self.tabs:
            # Switch to AI Analysis tab
            for i, tab_name in enumerate(['filter', 'analysis', 'view', 'action_potential', 'ai_analysis']):
                if tab_name == 'ai_analysis':
                    self.notebook.select(i)
                    break
            # Trigger AI analysis
            self.tabs['ai_analysis'].run_ai_analysis()
        else:
            messagebox.showwarning("AI Not Available", "AI Analysis is not available in this installation.")

    def run_manual_analysis(self):
        """Switch to AI tab and run manual analysis"""
        if hasattr(self, 'tabs') and 'ai_analysis' in self.tabs:
            # Switch to AI Analysis tab
            for i, tab_name in enumerate(['filter', 'analysis', 'view', 'action_potential', 'ai_analysis']):
                if tab_name == 'ai_analysis':
                    self.notebook.select(i)
                    break
            # Trigger manual analysis
            self.tabs['ai_analysis'].run_manual_analysis()
        else:
            messagebox.showwarning("AI Not Available", "AI Analysis is not available in this installation.")

    def validate_ai_results(self):
        """Switch to AI tab and run validation"""
        if hasattr(self, 'tabs') and 'ai_analysis' in self.tabs:
            # Switch to AI Analysis tab
            for i, tab_name in enumerate(['filter', 'analysis', 'view', 'action_potential', 'ai_analysis']):
                if tab_name == 'ai_analysis':
                    self.notebook.select(i)
                    break
            # Trigger validation
            self.tabs['ai_analysis'].validate_results()
        else:
            messagebox.showwarning("AI Not Available", "AI Analysis is not available in this installation.")

    def setup_main_layout(self):
        """Setup the main application layout with status bar"""
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
        
        # Create status bar at the bottom
        self.status_bar_frame = ttk.Frame(self.master, relief=tk.SUNKEN, border=1)
        self.status_bar_frame.pack(side='bottom', fill='x')
        
        # Status bar style
        self.style = ttk.Style()
        self.style.configure(
            "StatusBar.TLabel", 
            font=('Consolas', 9),  # Fixed-width font
            background="#f0f0f0",  # Light gray background
            foreground="#000000",  # Black text
            padding=(5, 2)         # Padding
        )
        
        # Status variable for point tracking
        self.point_status_var = tk.StringVar(value="No data loaded")
        
        # Create status label with fixed width font for better alignment
        self.point_status_label = ttk.Label(
            self.status_bar_frame, 
            textvariable=self.point_status_var,
            style="StatusBar.TLabel"
        )
        self.point_status_label.pack(fill='x', expand=True, anchor='w')

    def debug_point_tracking(self):
        """Debug the point tracking system and force-enable it if needed."""
        try:
            # Check point tracker status
            has_tracker = hasattr(self, 'point_tracker')
            
            # Build diagnostic message with relevant information
            message = f"Point Tracker Status:\n"
            message += f"- Point tracker exists: {has_tracker}\n"
            
            if has_tracker:
                # Check show_points flag
                message += f"- Show points enabled: {self.point_tracker.show_points}\n"
                
                # Count curves with data
                active_curves = 0
                for curve_type, curve_data in self.point_tracker.curve_data.items():
                    has_data = curve_data['data'] is not None and len(curve_data['data']) > 0
                    is_visible = curve_data['visible']
                    message += f"- {curve_type} curve: {'✓' if has_data else '✗'} (visible: {is_visible})\n"
                    if has_data and is_visible:
                        active_curves += 1
                
                message += f"- Active curves: {active_curves}\n"
                
            # Check processor status
            processor_exists = hasattr(self, 'action_potential_processor')
            processor_valid = processor_exists and self.action_potential_processor is not None
            
            # Check show_points setting in UI
            has_ui_control = hasattr(self.action_potential_tab, 'show_points')
            ui_points_enabled = has_ui_control and self.action_potential_tab.show_points.get()
            
            message += f"- Processor exists: {processor_exists}\n"
            message += f"- UI show_points control: {has_ui_control} (value: {ui_points_enabled if has_ui_control else 'N/A'})\n"
            
            # Show diagnostic
            from tkinter import messagebox
            result = messagebox.askyesno("Point Tracking Debug", message + "\n\nWould you like to force-enable point tracking?")
            
            # Force-enable if requested
            if result:
                if has_tracker:
                    # Enable point tracking
                    self.point_tracker.set_show_points(True)
                    
                    # Update UI control if it exists
                    if has_ui_control:
                        self.action_potential_tab.show_points.set(True)
                    
                    # Refresh curve data
                    if processor_valid:
                        self.update_point_tracking(True)
                        
                    messagebox.showinfo("Point Tracking", "Point tracking has been force-enabled.")
                else:
                    messagebox.showerror("Error", "Cannot enable point tracking - tracker not initialized.")
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Debug Error", f"Error diagnosing point tracking: {str(e)}")

    def debug_processor(self):
        """Debug the processor status"""
        try:
            # Basic processor status checks
            has_processor = hasattr(self, 'action_potential_processor')
            processor_valid = has_processor and self.action_potential_processor is not None
            
            # Build diagnostic message
            message = f"Processor exists: {has_processor}\n"
            message += f"Processor is valid: {processor_valid}\n"
            
            if processor_valid:
                processor = self.action_potential_processor
                message += f"Has orange_curve: {hasattr(processor, 'orange_curve')}\n"
                message += f"Has modified_hyperpol: {hasattr(processor, 'modified_hyperpol')}\n"
                message += f"Has modified_depol: {hasattr(processor, 'modified_depol')}\n"
            
            # Show diagnostic
            messagebox.showinfo("Processor Debug", message)
        except Exception as e:
            messagebox.showerror("Debug Error", str(e))

    def setup_toolbar(self):
        """Setup the toolbar with file operations"""
        # File operations frame
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

        # Add the new "Export Sets" button HERE, to the export_frame
        self.set_export_button = add_set_export_to_toolbar(self, export_frame)

        # Add Multi-File Analysis button
        self.multi_file_button = add_multi_file_analysis_to_toolbar(self, export_frame)

        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        
        # Plots frame
        plots_frame = ttk.Frame(self.toolbar_frame)
        plots_frame.pack(side='left', fill='x')
        
        self.separate_plots_btn = ttk.Button(
            plots_frame, 
            text="Separate Plots ▼",
            command=self.show_plot_menu
        )
        self.separate_plots_btn.pack(side='left', padx=2)
        
        # Updates frame
        updates_frame = ttk.Frame(self.toolbar_frame)
        updates_frame.pack(side='left', padx=5)
        
        # Add Check for Updates button
        ttk.Button(
            updates_frame,
            text="Check for Updates",
            command=self.check_for_updates
        ).pack(side='left', padx=2)
        
        # Create the plot menu
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
        
        # Create the menu bar if it doesn't exist
        if not hasattr(self, 'menu_bar'):
            self.menu_bar = tk.Menu(self.master)
            self.master.config(menu=self.menu_bar)
            
            # Add Help menu
            help_menu = tk.Menu(self.menu_bar, tearoff=0)
            self.menu_bar.add_cascade(label="Help", menu=help_menu)
            
            # Add Check for Updates option
            help_menu.add_command(label="Check for Updates", command=self.check_for_updates)
        
        # Status label
        self.status_var = tk.StringVar(value="No data loaded")
        self.status_label = ttk.Label(self.toolbar_frame, textvariable=self.status_var)
        self.status_label.pack(side='right', padx=5)

        # In the setup_toolbar method of SignalAnalyzerApp
        debug_button = ttk.Button(self.toolbar_frame, 
                                  text="Debug Processor",
                                  command=self.debug_processor)
        debug_button.pack(side='left', padx=2)

    def check_for_updates(self):
        """Manually check for updates"""
        if hasattr(self, 'updater'):
            self.updater.silent = False  # Force showing dialogs for manual check
            self.updater.start_update_process()
        else:
            messagebox.showinfo("Updates", "Update system not initialized")

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

    def setup_hot_reload(self):
        """Setup hot reload system for development."""
        try:
            import os
            # Get project root directory (assuming main app is in project root)
            project_root = os.path.dirname(os.path.abspath(__file__))
            
            # Initialize hot reload with callback to refresh components
            success = initialize_hot_reload(
                project_root=project_root,
                callback=self.on_code_reloaded
            )
            
            if success:
                app_logger.info("🔥 Hot reload enabled - modify code without restarting!")
            else:
                app_logger.warning("Hot reload failed to initialize")
                
        except Exception as e:
            app_logger.error(f"Hot reload setup failed: {e}")
    
    def on_code_reloaded(self):
        """Called after code is hot reloaded - refresh components if needed."""
        try:
            # Refresh curve fitting manager if it exists
            if hasattr(self, 'curve_fitting_panel'):
                app_logger.info("Code reloaded - curve fitting refreshed")
            
            # Add any other component refreshes here
            app_logger.info("✅ Hot reload complete")
            
        except Exception as e:
            app_logger.error(f"Error in reload callback: {e}")

    def save_current_plot_limits(self):
        """Save current plot axis limits for restoration."""
        try:
            if hasattr(self, 'ax'):
                self.saved_plot_limits['xlim'] = self.ax.get_xlim()
                self.saved_plot_limits['ylim'] = self.ax.get_ylim()
                app_logger.debug(f"Saved plot limits: x={self.saved_plot_limits['xlim']}, y={self.saved_plot_limits['ylim']}")
        except Exception as e:
            app_logger.debug(f"Error saving plot limits: {e}")

    def restore_plot_limits(self):
        """Restore previously saved plot axis limits."""
        try:
            if (hasattr(self, 'ax') and self.saved_plot_limits['auto_restore'] and 
                self.saved_plot_limits['xlim'] is not None and 
                self.saved_plot_limits['ylim'] is not None):
                
                self.ax.set_xlim(self.saved_plot_limits['xlim'])
                self.ax.set_ylim(self.saved_plot_limits['ylim'])
                app_logger.debug(f"Restored plot limits: x={self.saved_plot_limits['xlim']}, y={self.saved_plot_limits['ylim']}")
                return True
        except Exception as e:
            app_logger.debug(f"Error restoring plot limits: {e}")
        return False

    def toggle_auto_restore_limits(self, enable=None):
        """Toggle automatic restoration of plot limits."""
        if enable is None:
            self.saved_plot_limits['auto_restore'] = not self.saved_plot_limits['auto_restore']
        else:
            self.saved_plot_limits['auto_restore'] = enable
        
        app_logger.info(f"Auto-restore plot limits: {self.saved_plot_limits['auto_restore']}")
        return self.saved_plot_limits['auto_restore']

    def clear_saved_limits(self):
        """Clear saved plot limits."""
        self.saved_plot_limits['xlim'] = None
        self.saved_plot_limits['ylim'] = None
        app_logger.debug("Cleared saved plot limits")

    def setup_plot_context_menu(self):
        """Setup right-click context menu for plot state management."""
        try:
            import tkinter as tk
            
            def on_right_click(event):
                if event.inaxes != self.ax:
                    return
                
                # Create context menu
                context_menu = tk.Menu(self.master, tearoff=0)
                
                # Add auto-restore toggle
                auto_restore_text = "Disable Auto-Zoom Restore" if self.saved_plot_limits['auto_restore'] else "Enable Auto-Zoom Restore"
                context_menu.add_command(
                    label=auto_restore_text,
                    command=lambda: self.toggle_auto_restore_limits()
                )
                
                context_menu.add_separator()
                
                # Add manual save/restore options
                context_menu.add_command(
                    label="Save Current Zoom",
                    command=self.save_current_plot_limits
                )
                
                context_menu.add_command(
                    label="Restore Saved Zoom",
                    command=lambda: [self.restore_plot_limits(), self.canvas.draw_idle()]
                )
                
                context_menu.add_command(
                    label="Clear Saved Zoom",
                    command=self.clear_saved_limits
                )
                
                # Show menu
                try:
                    context_menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
                except:
                    pass
                finally:
                    context_menu.grab_release()
            
            # Connect right-click event
            self.canvas.mpl_connect('button_press_event', 
                lambda event: on_right_click(event) if event.button == 3 else None)
            
            app_logger.debug("Plot context menu for zoom state management enabled")
            
        except Exception as e:
            app_logger.error(f"Error setting up plot context menu: {e}")

    def show_history(self):
        """Show the analysis history window."""
        try:
            history_window = HistoryWindow(self.master, self.history_manager)
            history_window.transient(self.master)
            history_window.grab_set()
        except Exception as e:
            app_logger.error(f"Error showing history window: {str(e)}")
            messagebox.showerror("Error", f"Failed to show history: {str(e)}")

    def on_show_points_toggle(self):
        """Handle toggling of point display in status bar"""
        show_points = self.show_points.get()
        app_logger.debug(f"Show points toggled: {show_points}")
        
        # Pass the setting to the main application
        params = self.get_parameters()
        params['show_points'] = show_points
        params['visibility_update'] = True  # Flag as visibility update only
        
        # Call callback with updated parameters
        self.update_callback(params)
        
        # Explicitly update the point tracker if we can access it directly
        try:
            # Get the main app reference
            app = self.parent.master
            if hasattr(app, 'point_tracker'):
                app.point_tracker.set_show_points(show_points)
                app_logger.debug(f"Directly updated point tracker show_points to {show_points}")
        except Exception as e:
            app_logger.error(f"Error updating point tracker directly: {str(e)}")

    def update_point_tracking(self, enable_annotations=False):
        """
        Update point tracking data and optionally enable annotations.
        Point tracking itself is always enabled.
        """
        if not hasattr(self, 'point_tracker'):
            return
            
        app_logger.debug(f"Updating point tracking (annotations: {enable_annotations})")
        
        # Set annotation visibility flag - point tracking itself is always enabled
        self.point_tracker.show_annotations = enable_annotations
        
        # Get processor for data
        processor = getattr(self, 'action_potential_processor', None)
        if processor is None:
            app_logger.warning("No processor available for point tracking")
            return
        
        # Set all curve data if available
        if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
            self.point_tracker.curve_data['orange']['data'] = processor.orange_curve
            self.point_tracker.curve_data['orange']['times'] = getattr(processor, 'orange_curve_times', None)
        
        if hasattr(processor, 'normalized_curve') and processor.normalized_curve is not None:
            self.point_tracker.curve_data['blue']['data'] = processor.normalized_curve
            self.point_tracker.curve_data['blue']['times'] = getattr(processor, 'normalized_curve_times', None)
        
        if hasattr(processor, 'average_curve') and processor.average_curve is not None:
            self.point_tracker.curve_data['magenta']['data'] = processor.average_curve
            self.point_tracker.curve_data['magenta']['times'] = getattr(processor, 'average_curve_times', None)
        
        if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
            self.point_tracker.curve_data['purple_hyperpol']['data'] = processor.modified_hyperpol
            self.point_tracker.curve_data['purple_hyperpol']['times'] = getattr(processor, 'modified_hyperpol_times', None)
        
        if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
            self.point_tracker.curve_data['purple_depol']['data'] = processor.modified_depol
            self.point_tracker.curve_data['purple_depol']['times'] = getattr(processor, 'modified_depol_times', None)
        
        # Always ensure event connections are active
        self.point_tracker._connect()
        
        # Clear annotations if they're being disabled
        if not enable_annotations:
            self.point_tracker.clear_annotations()

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

    def on_remove_spikes(self):
        """
        Remove spikes from the current action_potential_processor.
        Then re-run analysis or update plot to reflect changes.
        """
        try:
            # Check if processor exists
            if not hasattr(self, 'action_potential_processor') or self.action_potential_processor is None:
                messagebox.showwarning(
                    "No Processor", 
                    "Please load data and run analysis first."
                )
                return
            
            # Check if processor has necessary data
            processor = self.action_potential_processor
            if not hasattr(processor, 'orange_curve') or processor.orange_curve is None:
                messagebox.showwarning(
                    "No Data", 
                    "The processor doesn't have any curve data to process."
                )
                return
            
            # Execute spike removal
            remove_spikes_from_processor(processor)
            print(f"Processed spike removal on processor: {processor}")
            
            # Refresh the analysis with empty parameters to update the plot
            self.on_action_potential_analysis({})
            
            # Show success message
            messagebox.showinfo("Spike Removal", "Spikes removed successfully.")
            
        except Exception as e:
            import traceback
            print(f"Error in on_remove_spikes: {str(e)}")
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to remove spikes: {str(e)}")

    def plot_regression_lines(self, data, times, interval, color='purple', alpha=0.7):
        """Plot regression line for a given segment of data with improved visibility."""
        if data is None or times is None or len(data) < 2:
            return

        try:
            start_idx, end_idx = interval
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(data):
                return

            # Extract data for regression
            x = times[start_idx:end_idx+1]
            y = data[start_idx:end_idx+1]
            
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
        """Setup the matplotlib plot area with point tracking"""
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
        
        # Initialize point tracker with status bar variable
        from src.utils.point_counter import CurvePointTracker
        self.point_tracker = CurvePointTracker(self.fig, self.ax, self.point_status_var)
        self.setup_plot_context_menu()
        app_logger.info("Plot setup complete with zoom state preservation")

    def setup_tabs(self):
        """Setup the control tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs dictionary to store tab references
        self.tabs = {}
        
        # Create tabs
        self.filter_tab = FilterTab(self.notebook, self.on_filter_change)
        self.analysis_tab = AnalysisTab(self.notebook, self.on_analysis_update)
        self.view_tab = ViewTab(self.notebook, self.on_view_change)
        self.action_potential_tab = ActionPotentialTab(self.notebook, self.on_action_potential_analysis)
        
        # Store tab references
        self.tabs['filter'] = self.filter_tab
        self.tabs['analysis'] = self.analysis_tab
        self.tabs['view'] = self.view_tab
        self.tabs['action_potential'] = self.action_potential_tab
        
        # Add tabs to notebook
        self.notebook.add(self.filter_tab.frame, text='Filters')
        self.notebook.add(self.analysis_tab.frame, text='Analysis')
        self.notebook.add(self.view_tab.frame, text='View')
        self.notebook.add(self.action_potential_tab.frame, text='Action Potential')
        
        # Add AI Analysis tab with deferred import to prevent circular dependencies
        try:
            from src.gui.ai_analysis_tab import AIAnalysisTab
            self.tabs['ai_analysis'] = AIAnalysisTab(self.notebook, self)
            self.notebook.add(self.tabs['ai_analysis'].frame, text='AI Analysis')
            app_logger.info("Successfully loaded AI Analysis tab.")
        except Exception as e:
            app_logger.warning(f"AI Analysis tab not available: {e}")
            placeholder_frame = ttk.Frame(self.notebook)
            ttk.Label(placeholder_frame, text="AI Analysis module unavailable", justify='center').pack(padx=20, pady=20)
            self.notebook.add(placeholder_frame, text='AI Analysis', state='disabled')
        
        # Add Excel Learning tab with deferred import
        try:
            from src.gui.excel_learning_tab import ExcelLearningTab
            self.tabs['excel_learning'] = ExcelLearningTab(self.notebook, self)
            self.notebook.add(self.tabs['excel_learning'].frame, text='Excel Learning')
            app_logger.info("Successfully loaded Excel Learning tab.")
        except Exception as e:
            app_logger.warning(f"Excel Learning tab not available: {e}")
            placeholder_frame = ttk.Frame(self.notebook)
            ttk.Label(placeholder_frame, text="Excel Learning module unavailable", justify='center').pack(padx=20, pady=20)
            self.notebook.add(placeholder_frame, text='Excel Learning', state='disabled')
    
    def reset_point_tracker(self):
        """
        Completely reset the point tracker when loading new data or changing views.
        """
        if not hasattr(self, 'point_tracker'):
            return
            
        app_logger.debug("Resetting point tracker")
        
        try:
            # Clear all annotations and hide points
            self.point_tracker.clear_annotations()
            self.point_tracker._clear_status_display()
            
            # Reset curve data
            for curve_type in self.point_tracker.curve_data:
                self.point_tracker.curve_data[curve_type] = {
                    'data': None, 
                    'times': None, 
                    'visible': False
                }
            
            # Reset other state variables
            self.point_tracker.last_cursor_pos = None
            self.point_tracker.current_time = 0
            self.point_tracker.current_value = 0
            
            # Reconnect event handlers (in case they got disconnected)
            self.point_tracker._connect()
            
            # Update point tracking based on current show_points setting
            if hasattr(self.action_potential_tab, 'show_points'):
                show_points = self.action_potential_tab.show_points.get()
                self.point_tracker.set_show_points(show_points)
                
                if show_points and hasattr(self, 'action_potential_processor'):
                    # Refresh curve data from processor
                    self.update_point_tracking(show_points)
            
            app_logger.info("Point tracker reset successfully")
            
        except Exception as e:
            app_logger.error(f"Error resetting point tracker: {str(e)}")

    def _find_closest_point(self, x_ms, y_pA):
        """
        Find closest orange point when in All Points mode with improved precision.
        
        Args:
            x_ms: X coordinate in milliseconds
            y_pA: Y coordinate in picoamperes
            
        Returns:
            Tuple of (point_number, point_time) or (None, None) if no point found
        """
        # Check if we have orange curve data and are in all_points mode
        if (not hasattr(self, 'orange_curve') or self.orange_curve is None or 
            not hasattr(self, 'orange_curve_times') or self.orange_curve_times is None or 
            not hasattr(self.action_potential_tab, 'average_display_mode') or
            self.action_potential_tab.average_display_mode.get() != "all_points"):
            return None, None
                
        # Convert x_ms from milliseconds to seconds for comparison
        x_sec = x_ms / 1000.0
        
        # Find closest point index with vectorized operations
        diffs = np.abs(self.orange_curve_times - x_sec)
        closest_idx = np.argmin(diffs)
        min_diff = diffs[closest_idx]
        
        # Use adaptive thresholds based on data range
        x_range = np.max(self.orange_curve_times) - np.min(self.orange_curve_times)
        y_range = np.max(self.orange_curve) - np.min(self.orange_curve)
        
        # Scale thresholds based on data ranges (more generous)
        time_threshold = max(0.0005, 0.01 * x_range)  # 1% of time range or at least 0.5ms
        amp_threshold = max(10.0, 0.05 * y_range)     # 5% of amplitude range or at least 10 pA
        
        # Log detection details for debugging
        app_logger.debug(f"Point search: x={x_sec:.5f}s, closest={self.orange_curve_times[closest_idx]:.5f}s, diff={min_diff:.5f}/{time_threshold:.5f}")
        app_logger.debug(f"Amplitude: y={y_pA:.1f}, closest={self.orange_curve[closest_idx]:.1f}, diff={abs(self.orange_curve[closest_idx] - y_pA):.1f}/{amp_threshold:.1f}")
        
        # Check if closest point is within thresholds
        if min_diff <= time_threshold and abs(self.orange_curve[closest_idx] - y_pA) <= amp_threshold:
            point_number = closest_idx + 1  # 1-based indexing for display
            return point_number, self.orange_curve_times[closest_idx]
        
        return None, None

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
        """Load data from file with proper memory management."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return
                
            app_logger.info(f"Loading file: {filepath}")
            
            # Clear existing data using the clear method
            self.clear_all_data()

            self.clear_saved_limits()
            
            # Force cleanup before loading new data
            gc.collect()
            
            # Store current file path
            self.current_file = filepath
            
            # Load with memory optimization
            atf_handler = ATFHandler(filepath)
            atf_handler.load_atf()
            
            # Get data and store using properties
            new_time_data = atf_handler.get_column("Time")
            new_data = atf_handler.get_column("#1")
            
            self.time_data = new_time_data
            self.data = new_data
            self.filtered_data = new_data.copy()
            
            # Clean up handler
            del atf_handler
            gc.collect()
            
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
                self.action_potential_tab.V2.set(voltage)
            
            self.update_plot(force_full_range=True)
            filename = os.path.basename(filepath)
            self.status_var.set(f"Loaded: {filename}")
            
            # Log memory usage
            data_size_mb = (self.data.nbytes + self.time_data.nbytes) / 1024 / 1024
            app_logger.info(f"File loaded successfully. Data size: {data_size_mb:.1f} MB")
            
            # Check file history
            if hasattr(self, 'history_manager') and hasattr(self.history_manager, 'history_entries'):
                history = self.history_manager.history_entries
                file_history = [entry for entry in history if entry['filename'] == filename]
                
                if file_history:
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
            # Clean up on error
            self.clear_all_data()
            self.clear_saved_limits()

    def on_filter_change(self, filters):
        """Handle changes in filter settings with memory optimization"""
        if self.data is None:
            return
            
        try:
            app_logger.debug("Applying filters with memory optimization")
            self.current_filters = filters
            
            # Create temporary copy for filtering
            temp_data = self.data.copy()
            
            # Apply combined filter with memory optimization
            filtered_result = combined_filter(temp_data, **filters)
            
            # Store result and clean up temporary data
            self.filtered_data = filtered_result
            del temp_data
            gc.collect()
            
            self.update_plot()
            self.analysis_tab.update_filtered_data(self.filtered_data)
            
            self.window_manager.set_data(self.time_data, self.filtered_data)
            
            # Clear processor as data has changed
            self.action_potential_processor = None
            
        except Exception as e:
            app_logger.error(f"Error applying filters: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply filters: {str(e)}")
            # Reset to original data on error
            if self.data is not None:
                self.filtered_data = self.data.copy()

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
        # Add version tracking for debugging
        import os, time
        func_version = "1.0.1"  # Increment when modifying
        app_logger.debug(f"Running on_action_potential_analysis version {func_version}")
        
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
            self.clear_saved_limits()
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
                average_curve_times,
                force_full_range=True
            )

            # Update results in the UI
            self.action_potential_tab.update_results(results)
            
            # Make sure the action_potential_processor is fully stored
            app_logger.debug(f"Analysis complete - action_potential_processor reference is {self.action_potential_processor is not None}")
            
            # PASS THE PROCESSOR REFERENCE DIRECTLY to avoid lookup issues
            if hasattr(self.action_potential_tab, 'set_processor'):
                self.action_potential_tab.set_processor(self.action_potential_processor)

            try:
                # Update curve fitting data if available
                if (hasattr(self, 'curve_fitting_panel') and 
                    self.curve_fitting_panel and 
                    hasattr(self, 'action_potential_processor') and
                    self.action_potential_processor):
                    
                    self.curve_fitting_panel.update_curve_data()
                    app_logger.debug("Curve fitting data updated after analysis")
                    
            except Exception as e:
                app_logger.error(f"Error updating curve fitting data: {str(e)}")

            # --- ADD HISTORY ENTRY HERE ---
            # If we have a history manager and no errors, store the analysis results.
            if self.history_manager:
                self.history_manager.add_entry(
                    filename=self.current_file,
                    results=results,
                    analysis_type="manual"
                )
            
            # NEW CODE: Update point tracking with latest processor data
            # This ensures point tracking always works regardless of checkbox state
            if hasattr(self, 'point_tracker'):
                app_logger.debug("Updating point tracker with latest processor data")
                # Update without enabling annotations (just data tracking)
                self.update_point_tracking(False)
            
            app_logger.info("Action potential analysis completed successfully")

        except Exception as e:
            app_logger.error(f"Error in action potential analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            if hasattr(self.action_potential_tab, 'disable_points_ui'):
                self.action_potential_tab.disable_points_ui()

    def update_curve_fitting_after_processor_change(self):
        """Update curve fitting when processor data changes."""
        try:
            if (hasattr(self, 'curve_fitting_panel') and 
                self.curve_fitting_panel and
                hasattr(self, 'action_potential_processor')):
                
                self.curve_fitting_panel.update_curve_data()
                
        except Exception as e:
            app_logger.error(f"Error updating curve fitting after processor change: {str(e)}")

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
            self.fig.tight_layout(pad=2.0)
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

    def update_plot(self, view_params=None, force_full_range=False):
        """Update plot with memory-efficient rendering and preserved zoom state."""
        if self.data is None:
            return
            
        try:
            # Only save current limits if NOT forcing full range (i.e., not loading new data)
            if not force_full_range:
                self.save_current_plot_limits()
            
            # Clear previous plot to free memory
            self.ax.clear()
            
            # Get view parameters
            if view_params is None:
                view_params = self.view_tab.get_view_params()
            
            # Determine what to plot
            show_original = view_params.get('show_original', True)
            show_filtered = view_params.get('show_filtered', True)
            
            # Use downsampling for large datasets to save memory
            if show_original and self.data is not None:
                time_plot, data_plot = self._downsample_for_plot(self.time_data, self.data)
                self.ax.plot(time_plot, data_plot, 'b-', label='Original', alpha=0.7, linewidth=1)
                del time_plot, data_plot  # Clean up immediately
            
            if show_filtered and self.filtered_data is not None:
                time_plot, filtered_plot = self._downsample_for_plot(self.time_data, self.filtered_data)
                self.ax.plot(time_plot, filtered_plot, 'r-', label='Filtered', linewidth=2)
                del time_plot, filtered_plot  # Clean up immediately
            
            # Configure axes BEFORE setting limits
            self._configure_axes()
            
            # Configure grid
            self.ax.grid(True, alpha=0.3)
            
            # Handle axis limits with preservation
            limits_applied = False
            
            # Force full range for new data loading
            if force_full_range:
                # Show full data range when loading new data
                if self.time_data is not None and self.data is not None:
                    self.ax.set_xlim(self.time_data[0], self.time_data[-1])
                    self.ax.set_ylim(np.min(self.data) * 1.05, np.max(self.data) * 1.05)  # Add 5% padding
                    limits_applied = True
                    app_logger.debug("Applied full range limits for new data")
            elif view_params.get('use_interval'):
                self.ax.set_xlim(view_params['t_min'], view_params['t_max'])
                limits_applied = True
            
            if view_params.get('use_custom_ylim'):
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
                limits_applied = True
            
            # ONLY restore saved limits if no explicit limits were applied AND not forcing full range
            if not limits_applied and not force_full_range:
                restored = self.restore_plot_limits()
                if restored:
                    app_logger.debug("Plot zoom state preserved")
            
            # Add legend if multiple traces
            handles, labels = self.ax.get_legend_handles_labels()
            if len(handles) > 1:
                self.ax.legend(loc='best')
            
            # Optimize plot for memory
            self.fig.tight_layout(pad=2.0)
            
            # Draw with memory optimization
            self.canvas.draw_idle()
            
            # Force cleanup after plotting
            gc.collect()
            
        except Exception as e:
            app_logger.error(f"Error updating plot: {e}")

    def _downsample_for_plot(self, time_data, signal_data, max_points=10000):
        """Downsample data for plotting to reduce memory usage"""
        if len(signal_data) <= max_points:
            return time_data, signal_data
        
        # Calculate downsampling factor
        factor = len(signal_data) // max_points
        
        # Downsample using indexing to preserve peaks
        indices = np.arange(0, len(signal_data), factor)
        
        return time_data[indices], signal_data[indices]

    def _plot_curve_if_enabled(self, x_data, y_data, enabled, *plot_args, **plot_kwargs):
        """Helper method to plot curve if enabled with memory cleanup"""
        if enabled and y_data is not None:
            try:
                # Downsample for memory efficiency
                x_plot, y_plot = self._downsample_for_plot(x_data, y_data)
                self.ax.plot(x_plot, y_plot, *plot_args, **plot_kwargs)
                del x_plot, y_plot  # Immediate cleanup
            except Exception as e:
                app_logger.error(f"Error plotting curve: {str(e)}")

    def _plot_processed_curve(self, processed_data, time_data):
        """Plot processed curve with memory optimization"""
        try:
            display_mode = self.action_potential_tab.processed_display_mode.get()
            time_ms = time_data * 1000
            
            if display_mode in ["line", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, processed_data)
                self.ax.plot(time_plot, data_plot, 'g-', 
                        label='Processed Signal',
                        linewidth=1.5, alpha=0.7)
                del time_plot, data_plot
                
            if display_mode in ["points", "all_points"]:
                # For points, use more aggressive downsampling
                time_plot, data_plot = self._downsample_for_plot(time_ms, processed_data, max_points=5000)
                self.ax.scatter(time_plot, data_plot, 
                            color='green', s=15, alpha=0.8, 
                            marker='.', label='Processed Points')
                del time_plot, data_plot
                
        except Exception as e:
            app_logger.error(f"Error plotting processed curve: {str(e)}")

    def _plot_average_curve(self, orange_curve, orange_times):
        """Plot average curve with memory optimization"""
        try:
            display_mode = self.action_potential_tab.average_display_mode.get()
            time_ms = orange_times * 1000
            
            if display_mode in ["line", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, orange_curve)
                self.ax.plot(time_plot, data_plot, 
                        color='#FFA500', label='50-point Average', 
                        linewidth=1.5, alpha=0.7)
                del time_plot, data_plot
                
            if display_mode in ["points", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, orange_curve, max_points=5000)
                self.ax.scatter(time_plot, data_plot, 
                            color='#FFA500', s=25, alpha=1, 
                            marker='o', label='Average Points')
                del time_plot, data_plot
                
        except Exception as e:
            app_logger.error(f"Error plotting average curve: {str(e)}")

    def _plot_normalized_curve(self, normalized_curve, normalized_times):
        """Plot normalized curve with memory optimization"""
        try:
            display_mode = self.action_potential_tab.normalized_display_mode.get()
            time_ms = normalized_times * 1000
            
            if display_mode in ["line", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, normalized_curve)
                self.ax.plot(time_plot, data_plot, 
                        color='#0057B8', label='Voltage-Normalized', 
                        linewidth=1.5, alpha=0.7)
                del time_plot, data_plot
                
            if display_mode in ["points", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, normalized_curve, max_points=5000)
                self.ax.scatter(time_plot, data_plot, 
                            color='#0057B8', s=25, alpha=1,
                            marker='o', label='Normalized Points')
                del time_plot, data_plot
                
        except Exception as e:
            app_logger.error(f"Error plotting normalized curve: {str(e)}")

    def _plot_averaged_normalized_curve(self, average_curve, average_curve_times):
        """Plot averaged normalized curve with memory optimization"""
        try:
            display_mode = self.action_potential_tab.averaged_normalized_display_mode.get()
            time_ms = average_curve_times * 1000
            
            if display_mode in ["line", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, average_curve)
                self.ax.plot(time_plot, data_plot, 
                        color='magenta', label='Averaged Normalized', 
                        linewidth=2, alpha=0.8)
                del time_plot, data_plot
                
            if display_mode in ["points", "all_points"]:
                time_plot, data_plot = self._downsample_for_plot(time_ms, average_curve, max_points=5000)
                self.ax.scatter(time_plot, data_plot, 
                            color='magenta', s=30, alpha=1,
                            marker='o', label='Avg Normalized Points')
                del time_plot, data_plot
                
        except Exception as e:
            app_logger.error(f"Error plotting averaged normalized curve: {str(e)}")

    def _plot_purple_curves_with_ranges(self, display_options, integration_ranges, intervals, show_points):
        """Plot purple curves with integration ranges and memory optimization"""
        try:
            has_purple_curves = (hasattr(self.action_potential_processor, 'modified_hyperpol') and 
                            hasattr(self.action_potential_processor, 'modified_depol') and
                            self.action_potential_processor.modified_hyperpol is not None and 
                            self.action_potential_processor.modified_depol is not None)

            if not (has_purple_curves and display_options.get('show_modified', True)):
                return
                
            display_mode = self.action_potential_tab.modified_display_mode.get()
            
            # Get data
            hyperpol = self.action_potential_processor.modified_hyperpol
            hyperpol_times = self.action_potential_processor.modified_hyperpol_times
            depol = self.action_potential_processor.modified_depol
            depol_times = self.action_potential_processor.modified_depol_times
            
            # Plot curves based on display mode
            if display_mode in ["line", "all_points"]:
                # Use downsampling for memory efficiency
                hyp_time_plot, hyp_data_plot = self._downsample_for_plot(hyperpol_times * 1000, hyperpol)
                dep_time_plot, dep_data_plot = self._downsample_for_plot(depol_times * 1000, depol)
                
                self.ax.plot(hyp_time_plot, hyp_data_plot,
                        color='purple', label='Modified Peaks', 
                        linewidth=2, alpha=0.8)
                self.ax.plot(dep_time_plot, dep_data_plot,
                        color='purple', label='_nolegend_',
                        linewidth=2, alpha=0.8)
                        
                del hyp_time_plot, hyp_data_plot, dep_time_plot, dep_data_plot
                        
            if display_mode in ["points", "all_points"]:
                # More aggressive downsampling for scatter plots
                hyp_time_plot, hyp_data_plot = self._downsample_for_plot(hyperpol_times * 1000, hyperpol, max_points=3000)
                dep_time_plot, dep_data_plot = self._downsample_for_plot(depol_times * 1000, depol, max_points=3000)
                
                self.ax.scatter(hyp_time_plot, hyp_data_plot,
                            color='purple', s=30, alpha=0.8,
                            marker='o',
                            label='_nolegend_' if display_mode == "all_points" else "Modified Points")
                self.ax.scatter(dep_time_plot, dep_data_plot,
                            color='purple', s=30, alpha=0.8,
                            marker='o', label='_nolegend_')
                            
                del hyp_time_plot, hyp_data_plot, dep_time_plot, dep_data_plot
            
            # Add integration range visualizations
            self._add_integration_range_visualizations(integration_ranges, hyperpol_times, depol_times)
            
            # Add regression lines if enabled
            if show_points and hasattr(self, 'plot_regression_lines'):
                regression_interval = intervals.get('regression_interval')
                if regression_interval:
                    self.plot_regression_lines(hyperpol, hyperpol_times, regression_interval, color='blue', alpha=0.8)
                    self.plot_regression_lines(depol, depol_times, regression_interval, color='red', alpha=0.8)
                    
        except Exception as e:
            app_logger.error(f"Error plotting purple curves: {str(e)}")

    def _add_integration_range_visualizations(self, integration_ranges, hyperpol_times, depol_times):
        """Add integration range visualizations with memory optimization"""
        try:
            if not integration_ranges:
                return
                
            # Hyperpolarization range
            if 'hyperpol' in integration_ranges and hyperpol_times is not None:
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
            
            # Depolarization range
            if 'depol' in integration_ranges and depol_times is not None:
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
                    
        except Exception as e:
            app_logger.error(f"Error adding integration range visualizations: {str(e)}")

    def _update_span_selectors(self, show_points, integration_ranges):
        """Update span selector visibility and positions with memory considerations"""
        try:
            # Determine if span selectors should be active
            active_mode = False
            if hasattr(self.action_potential_tab, 'modified_display_mode'):
                active_mode = (self.action_potential_tab.modified_display_mode.get() in ["line", "all_points"])

            enable_spans = show_points and active_mode

            if hasattr(self, 'span_selector'):
                self.span_selector.set_visible(enable_spans)
                self.span_selector.set_active(enable_spans)

            # Update range span selectors if enabled
            if enable_spans and hasattr(self, 'hyperpol_span') and hasattr(self, 'depol_span'):
                self._position_span_selectors(integration_ranges)
            else:
                # Disable span selectors
                if hasattr(self, 'hyperpol_span'):
                    self.hyperpol_span.set_visible(False)
                    self.hyperpol_span.set_active(False)
                if hasattr(self, 'depol_span'):
                    self.depol_span.set_visible(False)
                    self.depol_span.set_active(False)
                    
        except Exception as e:
            app_logger.error(f"Error updating span selectors: {str(e)}")

    def _position_span_selectors(self, integration_ranges):
        """Position span selectors with memory considerations"""
        try:
            if not (hasattr(self.action_potential_processor, 'modified_hyperpol_times') and 
                   hasattr(self.action_potential_processor, 'modified_depol_times')):
                return
                
            hyperpol_times = self.action_potential_processor.modified_hyperpol_times
            depol_times = self.action_potential_processor.modified_depol_times
            
            # Position hyperpol span
            if 'hyperpol' in integration_ranges and hyperpol_times is not None:
                hyperpol_range = integration_ranges['hyperpol']
                start_idx = hyperpol_range['start']
                end_idx = hyperpol_range['end']
                
                if (0 <= start_idx < len(hyperpol_times) and 0 < end_idx <= len(hyperpol_times)):
                    new_extents = (
                        hyperpol_times[start_idx] * 1000, 
                        hyperpol_times[end_idx - 1] * 1000
                    )
                    
                    if not hasattr(self, 'prev_hyperpol_extents') or self.prev_hyperpol_extents != new_extents:
                        self.hyperpol_span.extents = new_extents
                        self.prev_hyperpol_extents = new_extents
                        
                    self.hyperpol_span.set_visible(True)
                    self.hyperpol_span.set_active(True)
            
            # Position depol span
            if 'depol' in integration_ranges and depol_times is not None:
                depol_range = integration_ranges['depol']
                start_idx = depol_range['start']
                end_idx = depol_range['end']
                
                if (0 <= start_idx < len(depol_times) and 0 < end_idx <= len(depol_times)):
                    new_extents = (
                        depol_times[start_idx] * 1000, 
                        depol_times[end_idx - 1] * 1000
                    )
                    
                    if not hasattr(self, 'prev_depol_extents') or self.prev_depol_extents != new_extents:
                        self.depol_span.extents = new_extents
                        self.prev_depol_extents = new_extents
                        
                    self.depol_span.set_visible(True)
                    self.depol_span.set_active(True)
                    
        except Exception as e:
            app_logger.error(f"Error positioning span selectors: {str(e)}")

    def update_plot_with_processed_data(
        self,
        processed_data,
        orange_curve,
        orange_times,
        normalized_curve,
        normalized_times,
        average_curve,
        average_curve_times,
        force_full_range=False,
        force_auto_scale=False
    ):
        """Update the main plot with all processed data and curves with preserved zoom state."""
        try:
            if not hasattr(self, 'action_potential_tab'):
                return

            # Only save current limits if NOT forcing full range or auto-scaling
            if not force_full_range and not force_auto_scale:
                self.save_current_plot_limits()

            # Clear plot and force cleanup
            self.ax.clear()
            gc.collect()
            
            view_params = self.view_tab.get_view_params()
            display_options = self.action_potential_tab.get_parameters().get('display_options', {})

            # Get integration ranges and points visibility
            intervals = self.action_potential_tab.get_intervals()
            show_points = intervals.get('show_points', False)
            integration_ranges = intervals.get('integration_ranges', {})
            
            app_logger.debug(f"Updating plot with integration ranges: {integration_ranges}")

            # Plot curves with memory optimization
            self._plot_curve_if_enabled(
                self.time_data * 1000, self.data, 
                display_options.get('show_noisy_original', False),
                'b-', 'Original Signal', alpha=0.3
            )

            self._plot_curve_if_enabled(
                self.time_data * 1000, self.filtered_data,
                display_options.get('show_red_curve', True),
                '#800000', 'Filtered Signal', alpha=0.7, linewidth=1.5
            )

            # Plot processed curves
            if processed_data is not None and display_options.get('show_processed', True):
                self._plot_processed_curve(processed_data, self.time_data)

            # Plot other curves
            if orange_curve is not None and orange_times is not None and display_options.get('show_average', True):
                self._plot_average_curve(orange_curve, orange_times)

            if normalized_curve is not None and normalized_times is not None and display_options.get('show_normalized', True):
                self._plot_normalized_curve(normalized_curve, normalized_times)

            if average_curve is not None and average_curve_times is not None and display_options.get('show_averaged_normalized', True):
                self._plot_averaged_normalized_curve(average_curve, average_curve_times)

            # Plot purple curves with integration ranges
            self._plot_purple_curves_with_ranges(display_options, integration_ranges, intervals, show_points)

            # Configure axes with proper labels and ticks
            self._configure_axes()
            
            # Configure grid
            self.ax.grid(True, alpha=0.3)
            
            # Add legend
            self.ax.legend()

            # Apply view limits with improved logic
            limits_applied = False
            
            # Handle forced full range (for new data)
            if force_full_range and self.time_data is not None and self.data is not None:
                self.ax.set_xlim(self.time_data[0] * 1000, self.time_data[-1] * 1000)
                self.ax.set_ylim(np.min(self.data) * 1.05, np.max(self.data) * 1.05)
                limits_applied = True
                app_logger.debug("Applied full range limits for processed data display")
                
            # Handle auto-scaling for analysis results
            elif force_auto_scale:
                # Calculate appropriate limits based on visible data
                self._apply_auto_scale_limits(display_options)
                limits_applied = True
                app_logger.debug("Applied auto-scale limits for analysis results")
                
            # Apply explicit view parameters
            elif view_params.get('use_custom_ylim', False):
                self.ax.set_ylim(view_params['y_min'], view_params['y_max'])
                limits_applied = True
            elif view_params.get('use_interval', False):
                self.ax.set_xlim(view_params['t_min'] * 1000, view_params['t_max'] * 1000)
                limits_applied = True

            # ONLY restore saved limits if no explicit limits were applied and not forcing any scaling
            if not limits_applied and not force_full_range and not force_auto_scale:
                restored = self.restore_plot_limits()
                if restored:
                    app_logger.debug("Plot zoom state preserved")

            # Use tight layout with padding to prevent label cutoff
            self.fig.tight_layout(pad=2.0)
            self.canvas.draw_idle()
            
            # Handle span selectors
            self._update_span_selectors(show_points, integration_ranges)
            
            # Force cleanup after plotting
            gc.collect()
            
            app_logger.debug("Plot updated with preserved zoom state")

        except Exception as e:
            app_logger.error(f"Error updating plot with processed data: {e}")
            gc.collect()  # Cleanup on error
            raise

    def _apply_auto_scale_limits(self, display_options):
        """Apply intelligent auto-scaling based on visible data."""
        try:
            # Collect all visible data to determine appropriate limits
            all_y_data = []
            all_x_data = []
            
            # Collect data from visible curves
            if display_options.get('show_red_curve', True) and self.filtered_data is not None:
                all_y_data.extend(self.filtered_data)
                all_x_data.extend(self.time_data * 1000)
            
            if (display_options.get('show_processed', True) and 
                hasattr(self, 'processed_data') and self.processed_data is not None):
                all_y_data.extend(self.processed_data)
                all_x_data.extend(self.time_data * 1000)
            
            # Include orange curve
            if (display_options.get('show_average', True) and 
                hasattr(self, 'orange_curve') and self.orange_curve is not None):
                all_y_data.extend(self.orange_curve)
                all_x_data.extend(self.orange_curve_times * 1000)
            
            # Include normalized curves
            if (display_options.get('show_normalized', True) and 
                hasattr(self, 'normalized_curve') and self.normalized_curve is not None):
                all_y_data.extend(self.normalized_curve)
                all_x_data.extend(self.normalized_curve_times * 1000)
            
            # Include purple curves
            if (display_options.get('show_modified', True) and 
                hasattr(self, 'action_potential_processor') and self.action_potential_processor):
                
                if hasattr(self.action_potential_processor, 'modified_hyperpol'):
                    hyperpol = self.action_potential_processor.modified_hyperpol
                    hyperpol_times = self.action_potential_processor.modified_hyperpol_times
                    if hyperpol is not None and hyperpol_times is not None:
                        all_y_data.extend(hyperpol)
                        all_x_data.extend(hyperpol_times * 1000)
                
                if hasattr(self.action_potential_processor, 'modified_depol'):
                    depol = self.action_potential_processor.modified_depol
                    depol_times = self.action_potential_processor.modified_depol_times
                    if depol is not None and depol_times is not None:
                        all_y_data.extend(depol)
                        all_x_data.extend(depol_times * 1000)
            
            # Calculate limits with padding
            if all_y_data and all_x_data:
                y_min, y_max = np.min(all_y_data), np.max(all_y_data)
                x_min, x_max = np.min(all_x_data), np.max(all_x_data)
                
                # Add 5% padding
                y_range = y_max - y_min
                x_range = x_max - x_min
                
                y_padding = y_range * 0.05 if y_range > 0 else abs(y_max) * 0.1
                x_padding = x_range * 0.02 if x_range > 0 else abs(x_max) * 0.02
                
                self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
                
                app_logger.debug(f"Auto-scale applied: x=({x_min:.1f}, {x_max:.1f}), y=({y_min:.1f}, {y_max:.1f})")
            else:
                # Fallback to matplotlib auto-scaling
                self.ax.relim()
                self.ax.autoscale(True)
                app_logger.debug("Used matplotlib auto-scaling as fallback")
        
        except Exception as e:
            app_logger.error(f"Error in auto-scaling: {str(e)}")
            # Fallback to matplotlib auto-scaling
            try:
                self.ax.relim()
                self.ax.autoscale(True)
            except:
                pass

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

    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            app_logger.info("Application closing - performing cleanup")

            # Stop hot reload
            stop_hot_reload()

            # Cleanup Excel Learning tab if available
            if 'excel_learning' in self.tabs and hasattr(self.tabs['excel_learning'], 'cleanup'):
                try:
                    self.tabs['excel_learning'].cleanup()
                    app_logger.info("Excel Learning tab cleaned up")
                except Exception as e:
                    app_logger.warning(f"Error cleaning up Excel Learning tab: {e}")
            
            # Clear all data
            self.clear_all_data()
            
            # Close all matplotlib figures
            plt.close('all')
            
            # Force final garbage collection
            gc.collect()
            
            # Destroy the window
            self.master.quit()
            self.master.destroy()
            
        except Exception as e:
            app_logger.error(f"Error during application close: {str(e)}")
            # Force close even if cleanup fails
            self.master.quit()
            self.master.destroy()


# For standalone testing
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SignalAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        app_logger.critical(f"Application crashed: {str(e)}")
        raise