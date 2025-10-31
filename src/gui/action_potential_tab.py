import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger
import numpy as np
from src.gui.range_selection_utils import RangeSelectionManager
from src.gui.direct_spike_removal import remove_spikes_from_processor
import os, time
from src.gui.curve_fitting_gui import CurveFittingPanel
from src.analysis.curve_fitting_manager import CurveFittingManager
import tkinter as tk
print(f"action_potential_tab.py last modified: {time.ctime(os.path.getmtime(__file__))}")

class ActionPotentialTab:
    def __init__(self, parent, callback):
        """Initialize the action potential analysis tab."""
        self.parent = parent
        self.update_callback = callback
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Action Potential Analysis")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.curve_fitting_panel = None
        
        # Create canvas and scrollbar for scrolling
        self.canvas = tk.Canvas(self.frame, width=260)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar components
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Initialize variables
        self.init_variables()
        
        # Setup all UI components in correct order
        self.setup_results_display()  # First, setup results display
        self.setup_spike_removal_controls()
        self.setup_parameter_controls()
        self.setup_normalization_points()
        self.setup_integration_range_controls()  # Add integration range controls
        self.setup_regression_controls()
        self.setup_analysis_controls()
        
        # Configure mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Configure canvas resize
        self.frame.bind('<Configure>', self._on_frame_configure)

        self._analysis_results = {}
        
        app_logger.debug("Action potential analysis tab initialized")

    def _on_frame_configure(self, event=None):
        """Handle frame resizing"""
        width = max(260, self.frame.winfo_width() - 25)
        self.canvas.configure(width=width)
        self.canvas.itemconfig('window', width=width)

    def setup_spike_removal_controls(self):
        spike_frame = ttk.LabelFrame(self.scrollable_frame, text="Spike Removal")
        spike_frame.pack(fill='x', padx=5, pady=5)

        # Optional label/description
        ttk.Label(
            spike_frame,
            text="Eliminate periodic spikes at (n + 200*i) from all curves.",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

        remove_button = ttk.Button(
            spike_frame,
            text="Remove Spikes",
            command=self.on_remove_spikes_click
        )
        remove_button.pack(pady=5)

    def on_remove_spikes_click(self):
        """Handler for the 'Remove Spikes' button."""
        try:
            app_logger.debug("Remove spikes button clicked")
            
            # Get the processor from the main app
            app = self.parent.master  # The main SignalAnalyzerApp instance
            
            # First try to get the processor from the direct reference (most reliable)
            processor = getattr(self, 'processor', None)
            
            # If not available, try getting it from the app
            if processor is None:
                processor = getattr(app, 'action_potential_processor', None)
                
            # Check if processor exists and has necessary data
            if (not processor or 
                not hasattr(processor, 'orange_curve') or 
                processor.orange_curve is None):
                
                app_logger.warning("No valid processor available for spike removal")
                messagebox.showwarning(
                    "No Processor",
                    "Please load data and run analysis first."
                )
                return

            # Remove spikes from all curves
            remove_spikes_from_processor(processor)
            app_logger.info("Spikes removed from signal data")

            # Re-run analysis with empty parameters to update plots
            app.on_action_potential_analysis({})
            
            messagebox.showinfo("Spike Removal", "Spikes removed successfully.")

        except Exception as e:
            app_logger.error(f"Error removing spikes: {str(e)}")
            messagebox.showerror("Error", f"Failed to remove spikes: {str(e)}")

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_normalization_points(self):
        """Setup single input field for the starting point with simulation feature"""
        norm_frame = ttk.LabelFrame(self.scrollable_frame, text="Normalization Point")
        norm_frame.pack(fill='x', padx=5, pady=5)

        # Single point input
        point_frame = ttk.Frame(norm_frame)
        point_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(point_frame, text="Starting Point:").pack(side='left')
        norm_entry = ttk.Entry(point_frame, textvariable=self.norm_point, width=10)
        norm_entry.pack(side='left', padx=5)
        
        # Simulation button
        self.simulate_button = ttk.Button(
            point_frame, 
            text="Find Optimal Point", 
            command=self.run_starting_point_simulation,
            width=15
        )
        self.simulate_button.pack(side='left', padx=5)
        
        # Auto-optimization checkbox
        auto_opt_frame = ttk.Frame(norm_frame)
        auto_opt_frame.pack(fill='x', padx=5, pady=2)
        
        self.auto_optimize = tk.BooleanVar(value=True)
        auto_opt_check = ttk.Checkbutton(
            auto_opt_frame,
            text="Auto-optimize starting point",
            variable=self.auto_optimize,
            command=self.on_auto_optimize_change
        )
        auto_opt_check.pack(side='left')
        
        # Info label
        ttk.Label(
            norm_frame,
            text="Leave blank to use default value (35). Auto-optimization finds the best starting point automatically.",
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

    def get_normalization_points(self):
        """Get segments based on single starting point"""
        try:
            start_point = self.norm_point.get().strip()
            
            if not start_point:
                return None
                
            try:
                n = int(start_point)
                if n < 1:
                    messagebox.showwarning("Invalid Input", "Starting point must be positive")
                    return None
                    
                return {
                    'seg1': (n, n + 199),
                    'seg2': (n + 200, n + 399),
                    'seg3': (n + 400, n + 599),
                    'seg4': (n + 600, n + 799)
                }
                
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number")
                return None

        except Exception as e:
            app_logger.error(f"Error getting normalization points: {str(e)}")
            return None
        
    def setup_regression_controls(self):
        """Setup regression range selection controls."""
        reg_frame = ttk.LabelFrame(self.scrollable_frame, text="Range Selection")
        reg_frame.pack(fill='x', padx=5, pady=5)

        # Show points toggle with better connection to existing analysis
        points_frame = ttk.Frame(reg_frame)
        points_frame.pack(fill='x', padx=5, pady=2)

        self.show_points = tk.BooleanVar(value=False)
        self.points_checkbox = ttk.Checkbutton(
            points_frame,
            text="Enable Points & Regression",
            variable=self.show_points,
            command=self.on_show_points_change
        )
        self.points_checkbox.pack(pady=5)
        
        # Initially disable the checkbox until analysis is complete
        # Use State Flags correctly:
        # The ttk widgets use state flags instead of the traditional 'state' property
        # Flags are strings preceded by either '!' (to remove) or nothing (to add)
        # E.g., ['disabled'] to disable, ['!disabled'] to enable
        self.points_checkbox.state(['disabled'])
        app_logger.debug("Points checkbox initially disabled")

        # Help text
        ttk.Label(
            points_frame,
            text="Enable to show points and drag directly on the purple curves",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

    def update_after_analysis(self):
        """Update UI state after successful analysis."""
        try:
            # First check if we can access the processor through the app
            app = self.parent.master
            processor_exists = hasattr(app, 'action_potential_processor') and app.action_potential_processor is not None
            
            if processor_exists:
                # Check if purple curves exist
                has_purple_curves = (
                    hasattr(app.action_potential_processor, 'modified_hyperpol') and 
                    app.action_potential_processor.modified_hyperpol is not None and
                    len(app.action_potential_processor.modified_hyperpol) > 0 and
                    hasattr(app.action_potential_processor, 'modified_depol') and 
                    app.action_potential_processor.modified_depol is not None and
                    len(app.action_potential_processor.modified_depol) > 0
                )
                
                # Enable the points & regression checkbox if purple curves exist
                if has_purple_curves:
                    # First enable the checkbox (make it clickable)
                    self.enable_points_ui()
                    app_logger.debug("Points checkbox enabled - purple curves are available")
                else:
                    self.disable_points_ui()
                    app_logger.debug("Points checkbox disabled - no purple curves available")
            else:
                # Schedule another attempt after a short delay (200ms)
                app_logger.debug("action_potential_processor not available yet, retrying in 200ms")
                self.parent.after(200, self.update_after_analysis)
                
        except Exception as e:
            app_logger.error(f"Error in update_after_analysis: {str(e)}")
            self.disable_points_ui()

    def enable_points_ui(self):
        """Enable the Points & Regression UI elements."""
        try:
            if hasattr(self, 'show_points'):
                # Store the current state
                current_state = self.show_points.get()
                
                # Enable the checkbox - use correct ttk state management with flags
                if hasattr(self, 'points_checkbox'):
                    # The correct way is to add the '!disabled' flag which REMOVES the disabled state
                    self.points_checkbox.state(['!disabled'])
                    app_logger.debug("Points checkbox UI element enabled with ['!disabled'] state")
                    
                    # If it was previously enabled, ensure it stays enabled
                    if current_state:
                        self.show_points.set(True)
                        self.check_and_enable_points()
        except Exception as e:
            app_logger.error(f"Error enabling points UI: {str(e)}")

    def disable_points_ui(self):
        """Disable the Points & Regression UI elements."""
        try:
            if hasattr(self, 'points_checkbox'):
                # Add the 'disabled' flag to disable the checkbox
                self.points_checkbox.state(['disabled'])
                app_logger.debug("Points & regression UI disabled with ['disabled'] state")
        except Exception as e:
            app_logger.error(f"Error disabling points UI: {str(e)}")

    # Add this to the ActionPotentialTab class's check_and_enable_points method

    # ------------------------------------------
    # action_potential_tab.py (partial snippet)
    # ------------------------------------------

    def check_and_enable_points(self):
        """
        If analysis is available, enable the purple curves, range sliders, and
        ensure both the hyperpol and depol ranges are clearly visible.
        """
        try:
            # First try using the direct processor reference
            processor = getattr(self, 'processor', None)
            
            # If not available, try getting it from the app
            if processor is None:
                app = self.parent.master
                processor = getattr(app, 'action_potential_processor', None)
            
            # Check if we have valid purple curves
            if (processor is not None and 
                hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None and
                len(processor.modified_hyperpol) > 0 and
                hasattr(processor, 'modified_depol') and processor.modified_depol is not None and
                len(processor.modified_depol) > 0):
                
                # Make sure the display mode is set to show the purple "all_points"
                self.modified_display_mode.set("all_points")

                # Enable the range sliders
                if hasattr(self, 'range_manager'):
                    self.range_manager.enable_controls(True)
                    
                    # Auto-set good default ranges if they are too narrow
                    if (self.range_manager.hyperpol_end.get() - self.range_manager.hyperpol_start.get()) < 10:
                        self.range_manager.hyperpol_start.set(0)
                        self.range_manager.hyperpol_end.set(200)

                    if (self.range_manager.depol_end.get() - self.range_manager.depol_start.get()) < 10:
                        self.range_manager.depol_start.set(0)
                        self.range_manager.depol_end.set(200)

                    # Force an immediate range update and plot redraw
                    self.range_manager.update_range_display()
                    if hasattr(self.range_manager, 'calculate_range_integral'):
                        self.range_manager.calculate_range_integral()
                        
                    # Store the processor reference in range manager too
                    self.range_manager.processor = processor
                
                app_logger.debug("Points & regression enabled - purple curves verified")
                return True

            else:
                # If no purple curves, turn the checkbox off and warn
                self.show_points.set(False)
                if hasattr(self, 'range_manager'):
                    self.range_manager.enable_controls(False)
                    
                app_logger.warning("No purple curves available for points & regression")
                messagebox.showinfo(
                    "Analysis Required",
                    "Please run 'Analyze Signal' first to generate the purple curves."
                )
                return False
                
        except Exception as e:
            self.show_points.set(False)
            app_logger.error(f"Error enabling points: {str(e)}")
            return False

    def on_show_points_change(self):
        """
        Handle changes to show_points checkbox - only affects annotations
        and regression features, not basic point tracking.
        """
        # Get current state
        show_points = self.show_points.get()
        app_logger.info(f"Show points toggled: {show_points}")
        
        # Get main app reference
        app = self.parent.master
        
        # Send update with integration ranges
        ranges = {}
        if hasattr(self, 'range_manager'):
            ranges = self.range_manager.get_integration_ranges()
            # Enable/disable range sliders
            self.range_manager.enable_controls(show_points)
        
        # Build params for update
        params = {
            'integration_ranges': ranges,
            'show_points': show_points,  # This now only controls visual elements like spans
            'visibility_update': True
        }
        
        # Call update callback
        if self.update_callback:
            self.update_callback(params)
        
        # Toggle annotation display mode only
        if hasattr(app, 'point_tracker'):
            app.point_tracker.show_annotations = show_points
        
        # Set display mode to points when annotations are enabled
        if show_points:
            self.modified_display_mode.set("all_points")
            self.average_display_mode.set("all_points")
        
        # Update span selectors if needed
        if hasattr(app, 'toggle_span_selectors'):
            app.toggle_span_selectors(show_points)

    def enable_range_sliders(self, enable):
        """Enable or disable range sliders."""
        if hasattr(self, 'range_manager'):
            self.range_manager.enable_controls(enable)
        else:
            # Fallback to the old way if range manager not available
            state = 'normal' if enable else 'disabled'
            for slider in [self.hyperpol_start_slider, self.hyperpol_end_slider,
                        self.depol_start_slider, self.depol_end_slider]:
                if hasattr(self, slider):
                    slider.configure(state=state)

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

    def setup_curve_fitting_panel(self, fig, ax, main_app):
        """
        Initialize curve fitting panel with external figure/axes.
        Call this from main app after creating the plot.
        """
        from src.gui.curve_fitting_gui import CurveFittingPanel
        
        # Create the panel in the scrollable frame
        self.curve_fitting_panel = CurveFittingPanel(self.scrollable_frame, main_app)
        
        # Initialize with the provided figure and axes
        self.curve_fitting_panel.initialize_fitting_manager(fig, ax)
        
        # Store references
        self.fig = fig
        self.ax = ax
        self.main_app = main_app

    def update_curve_fitting_data(self):
        """Update curve fitting data after analysis."""
        if hasattr(self, 'curve_fitting_panel') and self.curve_fitting_panel:
            self.curve_fitting_panel.update_curve_data()

    def get_results_dict(self):
        """Get the current analysis results as a dictionary, including range manager integrals."""
        results = {}
        
        # Extract values from StringVars
        try:
            results['integral_value'] = self.integral_result.get()
            results['hyperpol_area'] = self.hyperpol_result.get()
            results['depol_area'] = self.depol_result.get()
            results['linear_capacitance'] = self.capacitance_result.get()
            results['status'] = self.status_text.get()
        except Exception as e:
            app_logger.error(f"Error getting results: {str(e)}")
        
        # Include any additional results stored previously
        if hasattr(self, '_analysis_results') and isinstance(self._analysis_results, dict):
            results.update(self._analysis_results)
        
        # Add integrals from range manager
        if hasattr(self, 'range_manager'):
            results = self.update_results_with_integrals(results)
        
        return results

    # Add this method to update results with integral values from range manager
    def update_results_with_integrals(self, results):
        """Update results dictionary with integral values from range manager."""
        if not hasattr(self, 'range_manager'):
            return results
            
        # Get current integrals from range manager
        integrals = self.range_manager.get_current_integrals()
        
        # Update results
        results['hyperpol_area'] = f"{integrals['hyperpol_integral']:.3f} pC"
        results['depol_area'] = f"{integrals['depol_integral']:.3f} pC"
        
        # Calculate capacitance
        if 'hyperpol_integral' in integrals and 'depol_integral' in integrals:
            voltage_diff = abs(self.params.get('V2', 0) - self.params.get('V0', -80))
            if voltage_diff > 0:
                capacitance = abs(integrals['hyperpol_integral'] - integrals['depol_integral']) / voltage_diff
                results['capacitance_nF'] = f"{capacitance:.3f} nF"
        
        return results

    def update_integration_values(self, integration_results):
        """Update integration values from curve fitting manager."""
        try:
            if not integration_results:
                return
            
            # Update hyperpol integration if available
            if 'hyperpol' in integration_results:
                hyperpol_data = integration_results['hyperpol']
                hyperpol_value = f"{hyperpol_data['integral']:.3f} pC"
                self.hyperpol_result.set(hyperpol_value)
                app_logger.info(f"Updated hyperpol integration: {hyperpol_value}")
            
            # Update depol integration if available
            if 'depol' in integration_results:
                depol_data = integration_results['depol']
                depol_value = f"{depol_data['integral']:.3f} pC"
                self.depol_result.set(depol_value)
                app_logger.info(f"Updated depol integration: {depol_value}")
            
            # Calculate total integral
            total_integral = 0
            if 'hyperpol' in integration_results:
                total_integral += integration_results['hyperpol']['integral']
            if 'depol' in integration_results:
                total_integral += integration_results['depol']['integral']
            
            if total_integral != 0:
                self.integral_result.set(f"{total_integral:.3f} pC")
                app_logger.info(f"Updated total integral: {total_integral:.3f} pC")
            
        except Exception as e:
            app_logger.error(f"Failed to update integration values: {str(e)}")

    def update_range_display(self):
        """Update the display of current integration ranges."""
        if not hasattr(self, 'range_display'):
            return
            
        ranges = self.get_integration_ranges()
        if ranges:
            hyperpol = ranges['hyperpol']
            depol = ranges['depol']
            self.range_display.config(
                text=f"Hyperpol: {hyperpol['start']}-{hyperpol['end']}\n"
                    f"Depol: {depol['start']}-{depol['end']}"
            )

    def get_display_options(self):
        """Get current display options."""
        try:
            return {
                'show_noisy_original': self.show_noisy_original.get(),
                'show_red_curve': self.show_red_curve.get(),
                'show_processed': self.show_processed.get(),
                'show_average': self.show_average.get(),
                'show_normalized': self.show_normalized.get(),
                'show_modified': self.show_modified.get(),
                'show_averaged_normalized': self.show_averaged_normalized.get(),
                'processed_mode': self.processed_display_mode.get(),
                'average_mode': self.average_display_mode.get(),
                'normalized_mode': self.normalized_display_mode.get(),
                'modified_mode': self.modified_display_mode.get(),
                'averaged_normalized_mode': self.averaged_normalized_display_mode.get()
            }
        except Exception as e:
            app_logger.error(f"Error getting display options: {str(e)}")
            return {
                'show_noisy_original': False,
                'show_red_curve': True,
                'show_processed': True,
                'show_average': True,
                'show_normalized': True,
                'show_modified': True,
                'show_averaged_normalized': True,
                'processed_mode': 'line',
                'average_mode': 'line',
                'normalized_mode': 'line',
                'modified_mode': 'line',
                'averaged_normalized_mode': 'line'
            }

    # This method should replace the existing get_integration_ranges
    def get_integration_ranges(self):
        """Get current integration ranges from the range manager."""
        if hasattr(self, 'range_manager'):
            return self.range_manager.get_integration_ranges()
        
        # Fallback if range manager not available
        return {
            'hyperpol': {
                'start': self.hyperpol_start.get() if hasattr(self, 'hyperpol_start') else 0,
                'end': self.hyperpol_end.get() if hasattr(self, 'hyperpol_end') else 200
            },
            'depol': {
                'start': self.depol_start.get() if hasattr(self, 'depol_start') else 0,
                'end': self.depol_end.get() if hasattr(self, 'depol_end') else 200
            }
        }
        
    def set_processor(self, processor):
        """Set a direct reference to the action_potential_processor."""
        try:
            # Store a direct reference to the processor
            self.processor = processor
            
            # Now enable the UI if the processor is valid
            if processor is not None:
                # Check if processor has purple curves
                has_purple_curves = (
                    hasattr(processor, 'modified_hyperpol') and 
                    processor.modified_hyperpol is not None and
                    len(processor.modified_hyperpol) > 0 and
                    hasattr(processor, 'modified_depol') and 
                    processor.modified_depol is not None and
                    len(processor.modified_depol) > 0
                )
                
                if has_purple_curves:
                    # Enable the UI
                    self.enable_points_ui()
                    app_logger.debug("Points UI enabled - processor reference received with purple curves")
                else:
                    app_logger.debug("Processor received but no purple curves found")
                    self.disable_points_ui()
                    
                # Pass processor reference to range manager if it exists
                if hasattr(self, 'range_manager'):
                    self.range_manager.processor = processor
                    app_logger.debug("Processor reference passed to range manager")
            else:
                self.disable_points_ui()
                app_logger.debug("Null processor reference received - UI disabled")
                        
            app_logger.debug(f"Processor reference set: {processor is not None}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error setting processor reference: {str(e)}")
            self.disable_points_ui()
            return False

    def update_analysis_state(self):
        """Update UI state based on analysis existence."""
        try:
            app = self.parent.master
            if app and hasattr(app, 'action_potential_processor'):
                processor = app.action_potential_processor
                if processor and hasattr(processor, 'modified_hyperpol'):
                    # Analysis exists, enable controls
                    self.show_points.set(True)
                    self.check_and_enable_points()
        except Exception as e:
            app_logger.error(f"Error updating analysis state: {str(e)}")

    def retry_show_points(self, show_points):
        """Retry enabling show points after a short delay."""
        app_logger.debug("Retrying show points toggle")
        if hasattr(self.parent, 'master') and hasattr(self.parent.master, 'action_potential_processor'):
            # Try again now that the processor should be available
            self.show_points.set(show_points)
            self.on_show_points_change()

    def get_intervals(self):
        """Get current interval settings"""
        try:
            # Get integration intervals for both hyperpol and depol
            integration_ranges = self.get_integration_ranges()

            # Get regression interval if points are shown
            show_points = self.show_points.get() if hasattr(self, 'show_points') else False

            return {
                'integration_ranges': integration_ranges,
                'show_points': show_points
            }
        except Exception as e:
            app_logger.error(f"Error getting intervals: {str(e)}")
            return {
                'integration_ranges': {
                    'hyperpol': {'start': 0, 'end': 200},
                    'depol': {'start': 0, 'end': 200}
                },
                'show_points': False
            }

    def on_method_change(self, *args):
        """Handle changes in integration method"""
        try:
            # Trigger re-analysis with new method if we have already analyzed
            if self.integral_result.get() != "---":
                app_logger.debug(f"Integration method changed, triggering reanalysis")
                self.analyze_signal()
        except Exception as e:
            app_logger.error(f"Error handling method change: {str(e)}")

    def init_variables(self):
        """Initialize control variables with validation"""
        try:
            # Analysis parameters
            self.n_cycles = tk.IntVar(value=2)
            self.t0 = tk.DoubleVar(value=20.0)
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=-100.0)
            self.V2 = tk.DoubleVar(value=-20.0)  

            # Integration method
            self.integration_method = tk.StringVar(value="traditional")
            
            # Results display
            self.integral_result = tk.StringVar(value="---")
            self.hyperpol_result = tk.StringVar(value="---")
            self.depol_result = tk.StringVar(value="---")
            self.capacitance_result = tk.StringVar(value="---")
            self.status_text = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar()
            
            # Display mode variables
            self.show_noisy_original = tk.BooleanVar(value=False)
            self.show_red_curve = tk.BooleanVar(value=True)
            self.show_processed = tk.BooleanVar(value=True)
            self.show_average = tk.BooleanVar(value=True)
            self.show_normalized = tk.BooleanVar(value=True)
            self.show_modified = tk.BooleanVar(value=True)
            self.show_averaged_normalized = tk.BooleanVar(value=True)
            
            # Display modes for each curve type
            self.processed_display_mode = tk.StringVar(value="line")
            self.average_display_mode = tk.StringVar(value="line")
            self.normalized_display_mode = tk.StringVar(value="line")
            self.modified_display_mode = tk.StringVar(value="line")
            self.averaged_normalized_display_mode = tk.StringVar(value="line")

            # Integration and regression intervals
            self.hyperpol_start = tk.IntVar(value=0)
            self.hyperpol_end = tk.IntVar(value=200)
            self.depol_start = tk.IntVar(value=0)
            self.depol_end = tk.IntVar(value=200)
            self.show_points = tk.BooleanVar(value=False)
            
            # Add validation traces
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            self.t0.trace_add("write", self.validate_time_constant)
            self.t1.trace_add("write", self.validate_time_constant)
            self.t2.trace_add("write", self.validate_time_constant)
            self.V2.trace_add("write", self.validate_voltage)
            self.integration_method.trace_add("write", self.on_method_change)
            
            # Normalization points
            self.norm_points = {
                'seg1_start': tk.StringVar(),
                'seg1_end': tk.StringVar(),
                'seg2_start': tk.StringVar(),
                'seg2_end': tk.StringVar(),
                'seg3_start': tk.StringVar(),
                'seg3_end': tk.StringVar(),
                'seg4_start': tk.StringVar(),
                'seg4_end': tk.StringVar()
            }
            
            # Single normalization point
            self.norm_point = tk.StringVar()
            
            # Auto-optimization setting
            self.auto_optimize = tk.BooleanVar(value=True)
            
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

    def setup_parameter_controls(self):
        """Setup parameter input controls"""
        try:
            param_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameters")
            param_frame.pack(fill='x', padx=5, pady=2)
            
            # Integration method selection
            method_frame = ttk.LabelFrame(param_frame, text="Integration Method")
            method_frame.pack(fill='x', padx=5, pady=(2, 5))
            
            ttk.Radiobutton(method_frame, text="Traditional Method",
                        variable=self.integration_method,
                        value="traditional").pack(anchor='w', padx=5, pady=2)
            
            ttk.Radiobutton(method_frame, text="Averaged Normalized Method",
                        variable=self.integration_method,
                        value="alternative").pack(anchor='w', padx=5, pady=2)
            
            # Method description
            ttk.Label(method_frame, 
                     text="Note: Alternative method uses averaged normalized curves",
                     font=('TkDefaultFont', 8, 'italic')).pack(anchor='w', padx=5, pady=(0, 2))
            
            # Number of cycles
            cycles_frame = ttk.Frame(param_frame)
            cycles_frame.pack(fill='x', padx=5, pady=1)
            ttk.Label(cycles_frame, text="Number of Cycles:").pack(side='left')
            ttk.Entry(cycles_frame, textvariable=self.n_cycles,
                    width=10).pack(side='right')
            
            # Time constants
            time_frame = ttk.Frame(param_frame)
            time_frame.pack(fill='x', padx=5, pady=1)
            time_frame.grid_columnconfigure(1, weight=1)
            time_frame.grid_columnconfigure(3, weight=1)
            time_frame.grid_columnconfigure(5, weight=1)
            
            # t0, t1, t2 controls
            ttk.Label(time_frame, text="t0 (ms):").grid(row=0, column=0, padx=2)
            ttk.Entry(time_frame, textvariable=self.t0, width=8).grid(row=0, column=1, padx=2)
            
            ttk.Label(time_frame, text="t1 (ms):").grid(row=0, column=2, padx=2)
            ttk.Entry(time_frame, textvariable=self.t1, width=8).grid(row=0, column=3, padx=2)
            
            ttk.Label(time_frame, text="t2 (ms):").grid(row=0, column=4, padx=2)
            ttk.Entry(time_frame, textvariable=self.t2, width=8).grid(row=0, column=5, padx=2)
            
            ttk.Label(time_frame, 
                     text="t0: baseline, t1: hyperpolarization, t2: depolarization",
                     font=('TkDefaultFont', 8, 'italic')).grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
            # Voltage levels
            volt_frame = ttk.Frame(param_frame)
            volt_frame.pack(fill='x', padx=5, pady=1)
            volt_frame.grid_columnconfigure(1, weight=1)
            volt_frame.grid_columnconfigure(3, weight=1)
            volt_frame.grid_columnconfigure(5, weight=1)
            
            # V0, V1, V2 controls
            ttk.Label(volt_frame, text="V0 (mV):").grid(row=0, column=0, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V0, width=8).grid(row=0, column=1, padx=2)
            
            ttk.Label(volt_frame, text="V1 (mV):").grid(row=0, column=2, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V1, width=8).grid(row=0, column=3, padx=2)
            
            ttk.Label(volt_frame, text="V2 (mV):").grid(row=0, column=4, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V2, width=8).grid(row=0, column=5, padx=2)
            
            ttk.Label(volt_frame, 
                     text="V0: baseline, V1: hyperpolarization, V2: depolarization",
                     font=('TkDefaultFont', 8, 'italic')).grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_integration_range_controls(self):
        """Setup integration range controls using the RangeSelectionManager."""
        # Create the range selection manager
        self.range_manager = RangeSelectionManager(
            self,
            self.scrollable_frame,
            self.on_integration_interval_change
        )
        
        # Add a reference to our show_points variable
        # So the range manager can access it for callbacks
        self.range_manager.parent_show_points = self.show_points

    def setup_results_display(self):
        """Setup the results display section."""
        results_frame = ttk.LabelFrame(self.scrollable_frame, text="Results")
        results_frame.pack(fill='x', padx=5, pady=5)

        # Main integral value
        integral_frame = ttk.Frame(results_frame)
        integral_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(integral_frame, text="Integral Value:", anchor='w').pack(side='left')
        ttk.Label(integral_frame, textvariable=self.integral_result).pack(side='right')

        # Purple curve integrals
        purple_frame = ttk.Frame(results_frame)
        purple_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(purple_frame, text="Purple Curves:", anchor='w').pack(side='left')
        purple_right = ttk.Frame(purple_frame)
        purple_right.pack(side='right')

        # Hyperpolarization (indented)
        hyperpol_frame = ttk.Frame(purple_right)
        hyperpol_frame.pack(fill='x')
        ttk.Label(hyperpol_frame, text="Hyperpol:", width=10, anchor='e').pack(side='left')
        ttk.Label(hyperpol_frame, textvariable=self.hyperpol_result).pack(side='right')

        # Depolarization (indented)
        depol_frame = ttk.Frame(purple_right)
        depol_frame.pack(fill='x')
        ttk.Label(depol_frame, text="Depol:", width=10, anchor='e').pack(side='left')
        ttk.Label(depol_frame, textvariable=self.depol_result).pack(side='right')

        # Linear capacitance
        cap_frame = ttk.Frame(results_frame)
        cap_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(cap_frame, text="Linear Capacitance:", anchor='w').pack(side='left')
        ttk.Label(cap_frame, textvariable=self.capacitance_result).pack(side='right')

        # Progress and status
        progress_frame = ttk.Frame(results_frame)
        progress_frame.pack(fill='x', padx=5, pady=2)
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress.pack(fill='x', pady=2)
        ttk.Label(progress_frame, textvariable=self.status_text).pack(anchor='w')

    def fix_canvas_sizing(self):
        """Fix canvas sizing to allow proper zooming."""
        if hasattr(self, 'canvas'):
            # Make canvas properly resizable
            self.canvas.get_tk_widget().pack_forget()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Update figure size on resize
            def on_canvas_resize(event=None):
                if event and event.width > 0 and event.height > 0:
                    # Convert pixels to inches (assuming 100 DPI)
                    width_inches = event.width / 100
                    height_inches = event.height / 100
                    
                    # Update figure size
                    if hasattr(self, 'fig'):
                        self.fig.set_size_inches(width_inches, height_inches, forward=False)
                        self.canvas.draw_idle()
            
            # Bind resize event
            self.canvas.get_tk_widget().bind('<Configure>', on_canvas_resize)

    def on_remove_spikes_click(self):
        """
        Handle the "Remove Spikes" button click with robust processor access.
        
        This method has been fixed to properly locate the action_potential_processor
        regardless of where it's stored in the application hierarchy.
        
        This version is compatible with the correct function signature for
        remove_spikes_from_processor.
        """
        app_logger.info("=== Starting Spike Removal Process ===")
        
        # Find the processor using multiple search strategies
        processor = None
        processor_location = "unknown"
        
        # Strategy 1: Check for processor attribute directly
        if hasattr(self, 'processor') and self.processor is not None:
            processor = self.processor
            processor_location = "self.processor"
            app_logger.debug("Found processor in self.processor")
        
        # Strategy 2: Direct access from self
        elif hasattr(self, 'action_potential_processor') and self.action_potential_processor is not None:
            processor = self.action_potential_processor
            processor_location = "self.action_potential_processor"
            app_logger.debug("Found processor in self.action_potential_processor")
        
        # Strategy 3: Access from parent app
        elif hasattr(self, 'parent') and hasattr(self.parent, 'master'):
            app = self.parent.master
            if hasattr(app, 'action_potential_processor') and app.action_potential_processor is not None:
                processor = app.action_potential_processor
                processor_location = "self.parent.master.action_potential_processor"
                app_logger.debug("Found processor in parent.master")
        
        # Strategy 4: Look for processor reference in attributes
        if processor is None:
            for attr_name in dir(self):
                if attr_name.endswith('processor'):
                    attr = getattr(self, attr_name)
                    if attr is not None and hasattr(attr, 'orange_curve'):
                        processor = attr
                        processor_location = f"self.{attr_name}"
                        app_logger.debug(f"Found processor in self.{attr_name}")
                        break
        
        # Strategy 5: Check if we have a params dict that might have processor
        if processor is None and hasattr(self, 'params') and isinstance(self.params, dict):
            if 'processor' in self.params and self.params['processor'] is not None:
                processor = self.params['processor']
                processor_location = "self.params['processor']"
                app_logger.debug("Found processor in params dict")
        
        # Additional debug information about application state
        app_logger.debug(f"ACTION POTENTIAL PROCESSOR SEARCH:")
        app_logger.debug(f"  Has self.processor: {hasattr(self, 'processor')}")
        app_logger.debug(f"  Has self.action_potential_processor: {hasattr(self, 'action_potential_processor')}")
        if hasattr(self, 'parent') and hasattr(self.parent, 'master'):
            app_logger.debug(f"  Has parent.master.action_potential_processor: {hasattr(self.parent.master, 'action_potential_processor')}")
        app_logger.debug(f"  Processor found: {processor is not None}")
        app_logger.debug(f"  Location: {processor_location}")
        
        # Check if we found a processor
        if processor is None:
            messagebox.showinfo("Information", "Please run analysis first")
            app_logger.warning("Cannot remove spikes: No data processed yet")
            return
        
        # Verify processor has required data
        if not hasattr(processor, 'orange_curve') or processor.orange_curve is None:
            messagebox.showinfo("Information", "No curve data available. Please run analysis first.")
            app_logger.warning("Cannot remove spikes: No curve data in processor")
            return
        
        try:
            # Step 1: Apply adaptive spike removal algorithm
            from src.gui.direct_spike_removal import remove_spikes_from_processor
            success, results = remove_spikes_from_processor(processor)
            
            if not success:
                error_msg = results.get("error", "Unknown error during spike removal")
                messagebox.showerror("Error", f"Failed to remove spikes: {error_msg}")
                app_logger.error(f"Spike removal failed: {error_msg}")
                return
            
            total_replaced = results.get("total_replaced", 0)
            if total_replaced == 0:
                messagebox.showinfo("Information", "No problematic spikes were detected")
                app_logger.info("No spikes detected that meet removal criteria")
                return
            
            # Step 2: Update the display using multiple strategies
            app_logger.info(f"Successfully removed {total_replaced} spikes. Updating display...")
            
            update_successful = False
            
            # Find a suitable update method - try the most common method names
            update_methods = [
                # Method name, object, whether it's a replot method
                ('update_plot', self, False),
                ('update_action_potential_plot', self, False),
                ('update_plot_with_processed_data', self, False),
                ('on_action_potential_analysis', self, True),
                ('refresh_plot', self, False),
                ('redraw_plot', self, False),
                ('plot_action_potential', self, False),
                ('draw_action_potential', self, False),
                
                # Check parent methods
                ('update_plot', self.parent, False),
                ('refresh_plot', self.parent, False)
            ]
            
            # Try each method until one works
            for method_name, obj, is_replot in update_methods:
                if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
                    try:
                        method = getattr(obj, method_name)
                        app_logger.debug(f"Trying update method: {obj.__class__.__name__}.{method_name}")
                        
                        # Different approach based on whether this is a replot method
                        if is_replot and hasattr(self, 'params'):
                            # This is a method that reruns the whole analysis
                            # Pass the current parameters to keep settings
                            if isinstance(self.params, dict):
                                method(self.params)
                            else:
                                # Fall back to empty dict
                                method({})
                        else:
                            # This is a simple update method, try without args first
                            try:
                                method()
                            except TypeError:
                                # If that fails, try with processor
                                method(processor=processor)
                        
                        update_successful = True
                        app_logger.info(f"Successfully updated plot using {obj.__class__.__name__}.{method_name}")
                        break
                    except Exception as e:
                        app_logger.debug(f"Update method {method_name} failed: {str(e)}")
            
            # If we found a processor but couldn't update, try direct canvas update
            if not update_successful and hasattr(self, 'canvas') and self.canvas is not None:
                try:
                    app_logger.debug("Trying direct canvas update")
                    self.canvas.draw()  # Force immediate redraw
                    update_successful = True
                    app_logger.info("Successfully updated using canvas.draw()")
                except Exception as e:
                    app_logger.debug(f"Canvas update failed: {str(e)}")
            
            # Inform user of the result
            if update_successful:
                messagebox.showinfo("Success", f"Successfully removed {total_replaced} spikes")
                app_logger.info("=== Spike Removal Process Completed Successfully ===")
            else:
                message = (f"Spikes were successfully removed ({total_replaced} total), "
                        f"but the plot couldn't be automatically updated.\n\n"
                        f"Please click 'Run Analysis' again to see the cleaned data.")
                messagebox.showwarning("Partial Success", message)
                app_logger.warning("Spikes removed but plot update failed")
        
        except Exception as e:
            app_logger.error(f"Error in spike removal process: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"An error occurred during spike removal: {str(e)}")

    def update_results(self, results):
        """Update displayed results with proper formatting."""
        try:
            if isinstance(results, dict):
                # Update main integral value
                if 'integral_value' in results:
                    self.integral_result.set(results['integral_value'])
                
                # Update hyperpolarization result
                if 'hyperpol_area' in results:
                    # Extract numeric value from string if it ends with ' pC'
                    hyperpol_str = str(results['hyperpol_area'])
                    if ' pC' in hyperpol_str:
                        hyperpol_val = float(hyperpol_str.replace(' pC', ''))
                    else:
                        hyperpol_val = float(hyperpol_str)
                    self.hyperpol_result.set(f"{hyperpol_val:.2f} pC")
                
                # Update depolarization result
                if 'depol_area' in results:
                    # Extract numeric value from string if it ends with ' pC'
                    depol_str = str(results['depol_area'])
                    if ' pC' in depol_str:
                        depol_val = float(depol_str.replace(' pC', ''))
                    else:
                        depol_val = float(depol_str)
                    self.depol_result.set(f"{depol_val:.2f} pC")
                
                # Update linear capacitance
                if 'capacitance_nF' in results:
                    # Extract numeric value from string if it ends with ' nF'
                    cap_str = str(results['capacitance_nF'])
                    if ' nF' in cap_str:
                        cap_val = float(cap_str.replace(' nF', ''))
                    else:
                        cap_val = float(cap_str)
                    self.capacitance_result.set(f"{cap_val:.2f} nF")
                
                # Update status
                self.status_text.set(results.get('status', 'Results updated'))
                
                # Store full results for later access
                self._analysis_results = results.copy()
                
            else:
                self.integral_result.set("Error")
                self.hyperpol_result.set("Error")
                self.depol_result.set("Error")
                self.capacitance_result.set("Error")
                self.status_text.set("Error in results")
                
        except Exception as e:
            app_logger.error(f"Error updating results: {str(e)}")
            self.integral_result.set("Error")
            self.hyperpol_result.set("Error")
            self.depol_result.set("Error")
            self.capacitance_result.set("Error")
            self.status_text.set(f"Error: {str(e)}")

    def update_slider_display(self, value, label, slider_type):
        """Update display label for a slider and handle range validation."""
        try:
            val = int(float(value))
            label.config(text=str(val))
            
            # Ensure end is always greater than start
            if slider_type.endswith('_start'):
                end_var = getattr(self, slider_type.replace('start', 'end'))
                if val >= end_var.get():
                    end_var.set(val + 1)
            elif slider_type.endswith('_end'):
                start_var = getattr(self, slider_type.replace('end', 'start'))
                if val <= start_var.get():
                    start_var.set(val - 1)
                    
            self.on_integration_interval_change()
            
        except Exception as e:
            app_logger.error(f"Error updating slider display: {str(e)}")
    
    def setup_analysis_controls(self):
        """Setup analysis controls and display options."""
        analysis_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis")
        analysis_frame.pack(fill='x', padx=5, pady=5)
        
        # Display options frame
        display_frame = ttk.LabelFrame(analysis_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        # Show/hide signal checkbuttons
        signals = [
            ("Show Processed Signal", self.show_processed),
            ("Show Red Curve", self.show_red_curve),
            ("Show 50-point Average", self.show_average),
            ("Show Voltage-Normalized", self.show_normalized),
            ("Show Averaged Normalized", self.show_averaged_normalized),
            ("Show Modified Peaks", self.show_modified)
        ]
        
        for text, var in signals:
            ttk.Checkbutton(display_frame, text=text,
                           variable=var,
                           command=self.on_display_change).pack(anchor='w', padx=5)
        
        # Display modes frame
        modes_frame = ttk.Frame(display_frame)
        modes_frame.pack(fill='x', padx=5, pady=5)
        
        # Create mode controls grid
        control_frames = ttk.Frame(modes_frame)
        control_frames.pack(fill='x', expand=True)
        control_frames.grid_columnconfigure(0, weight=1)
        control_frames.grid_columnconfigure(1, weight=1)
        
        # Setup mode controls for each signal type
        mode_configs = [
            ("Processed", self.processed_display_mode, 0, 0),
            ("Average", self.average_display_mode, 0, 1),
            ("Normalized", self.normalized_display_mode, 1, 0),
            ("Modified", self.modified_display_mode, 1, 1),
            ("Avg Normalized", self.averaged_normalized_display_mode, 2, 0)
        ]
        
        for title, var, row, col in mode_configs:
            frame = ttk.LabelFrame(control_frames, text=title)
            frame.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(frame, 
                               text=mode.replace('_', ' ').title(),
                               variable=var,
                               value=mode,
                               command=self.on_display_change).pack(anchor='w')
        
        # Analysis button
        self.analyze_button = ttk.Button(analysis_frame, 
                                       text="Analyze Signal",
                                       command=self.analyze_signal)
        self.analyze_button.pack(pady=5)
        
        # Fast-track mode button
        fast_track_frame = ttk.LabelFrame(analysis_frame, text="Fast Track Mode")
        fast_track_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(fast_track_frame, 
                 text="Automate workflow:\nFilters → Analysis → Fitting → Export",
                 font=('Arial', 8)).pack(pady=2)
        
        self.fast_track_button = ttk.Button(fast_track_frame,
                                          text="⚡ Fast Track",
                                          command=self.start_fast_track)
        self.fast_track_button.pack(pady=2)
        
        ttk.Button(fast_track_frame,
                  text="💾 Save Session Log",
                  command=self.save_session_log).pack(pady=2)

    def analyze_signal(self):
        """Perform signal analysis with current parameters."""
        try:
            # Disable button during analysis
            self.analyze_button.state(['disabled'])
            self.status_text.set("Analyzing...")
            self.progress_var.set(0)
            
            # Get parameters and normalization points
            params = self.get_parameters()
            norm_points = self.get_normalization_points()
            if norm_points:
                params['normalization_points'] = norm_points
            
            # Log analysis start if action logger is available
            main_app = self._get_main_app()
            if main_app and hasattr(main_app, 'action_logger') and main_app.action_logger:
                try:
                    main_app.action_logger.log_analysis_start(params)
                except Exception as e:
                    app_logger.warning(f"Failed to log analysis start: {e}")
            
            # Update progress
            self.progress_var.set(50)
            
            # Call callback with parameters
            self.update_callback(params)
            
            # Log analysis complete if action logger is available
            if main_app and hasattr(main_app, 'action_logger') and main_app.action_logger:
                try:
                    main_app.action_logger.log_analysis_complete()
                except Exception as e:
                    app_logger.warning(f"Failed to log analysis complete: {e}")
            
            # Reset UI
            self.progress_var.set(100)
            self.status_text.set("Analysis complete")
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error in signal analysis: {str(e)}")
            self.status_text.set(f"Error: {str(e)}")
            self.analyze_button.state(['!disabled'])
            messagebox.showerror("Analysis Error", str(e))

    def on_display_change(self):
        """Handle changes in display options."""
        try:
            params = self.get_parameters()
            params['visibility_update'] = True  # Flag as visibility-only update to preserve zoom
            self.update_callback(params)
        except Exception as e:
            app_logger.error(f"Error handling display change: {str(e)}")

    def on_integration_interval_change(self, *args):
        """Handle changes to integration interval sliders."""
        try:
            # Get current integration ranges
            ranges = self.get_integration_ranges()
            app_logger.debug(f"Integration ranges updated: {ranges}")
            
            # Update display text
            self.update_range_display()
            
            # Send to main app for plot update
            if self.update_callback:
                self.update_callback({
                    'integration_ranges': ranges,
                    'show_points': self.show_points.get(),
                    'visibility_update': True  # Flag as visibility-only update
                })
                app_logger.debug("Plot update triggered with new integration ranges")
            else:
                app_logger.warning("No update callback available for integration ranges")
                
        except Exception as e:
            app_logger.error(f"Error handling integration interval change: {str(e)}")

    def get_parameters(self):
        """Get current analysis parameters with validation."""
        try:
            params = {
                'n_cycles': self.n_cycles.get(),
                't0': self.t0.get(),
                't1': self.t1.get(),
                't2': self.t2.get(),
                'V0': self.V0.get(),
                'V1': self.V1.get(),
                'V2': self.V2.get(),
                'use_alternative_method': self.integration_method.get() == "alternative",
                'auto_optimize_starting_point': self.auto_optimize.get(),
                'integration_ranges': self.get_integration_ranges(),
                'display_options': {
                    'show_noisy_original': self.show_noisy_original.get(),
                    'show_red_curve': self.show_red_curve.get(),
                    'show_processed': self.show_processed.get(),
                    'show_average': self.show_average.get(),
                    'show_normalized': self.show_normalized.get(),
                    'show_modified': self.show_modified.get(),
                    'show_averaged_normalized': self.show_averaged_normalized.get(),
                    'processed_mode': self.processed_display_mode.get(),
                    'average_mode': self.average_display_mode.get(),
                    'normalized_mode': self.normalized_display_mode.get(),
                    'modified_mode': self.modified_display_mode.get(),
                    'averaged_normalized_mode': self.averaged_normalized_display_mode.get()
                }
            }
            
            # Validate parameters
            if params['n_cycles'] < 1:
                raise ValueError("Number of cycles must be positive")
            if params['t0'] <= 0 or params['t1'] <= 0 or params['t2'] <= 0:
                raise ValueError("Time constants must be positive")
            
            return params
            
        except Exception as e:
            app_logger.error(f"Error getting parameters: {str(e)}")
            raise

    def validate_voltage(self, *args):
        """Validate voltage values and trigger reanalysis."""
        try:
            v2 = self.V2.get()
            if abs(v2) > 100:
                messagebox.showwarning("Validation", "Voltage should be between -100mV and +100mV")
                self.V2.set(-20.0)
                return
            
            # If valid and already analyzed once, trigger reanalysis
            if self.integral_result.get() != "---":
                app_logger.debug(f"V2 changed to {v2}mV, triggering reanalysis")
                self.analyze_signal()
                
        except tk.TclError:
            self.V2.set(-20.0)

    def validate_time_constant(self, *args):
        """Validate time constants."""
        try:
            t1, t2 = self.t1.get(), self.t2.get()
            if t1 <= 0 or t2 <= 0:
                messagebox.showwarning("Validation", "Time constants must be positive")
                if t1 <= 0: self.t1.set(100.0)
                if t2 <= 0: self.t2.set(100.0)
        except tk.TclError:
            self.t1.set(100.0)
            self.t2.set(100.0)

    def validate_n_cycles(self, *args):
        """Validate number of cycles."""
        try:
            value = self.n_cycles.get()
            if value < 1:
                self.n_cycles.set(1)
                messagebox.showwarning("Validation", "Number of cycles must be at least 1")
            elif value > 10:
                self.n_cycles.set(10)
                messagebox.showwarning("Validation", "Maximum number of cycles is 10")
        except tk.TclError:
            self.n_cycles.set(2)

    def reset(self):
        """Reset the tab to initial state."""
        try:
            # Reset parameters
            self.n_cycles.set(2)
            self.t1.set(100.0)
            self.t2.set(100.0)
            self.V0.set(-80.0)
            self.V1.set(-100.0)
            self.V2.set(-20.0)
            
            # Reset results
            self.integral_result.set("---")
            self.hyperpol_result.set("---")
            self.depol_result.set("---")
            self.capacitance_result.set("---")
            self.status_text.set("Ready")
            self.progress_var.set(0)
            
            # Reset display options
            self.show_processed.set(True)
            self.show_red_curve.set(True)
            self.show_average.set(True)
            self.show_normalized.set(True)
            self.show_modified.set(True)
            self.show_averaged_normalized.set(True)
            
            # Reset display modes
            for var in [self.processed_display_mode, self.average_display_mode,
                       self.normalized_display_mode, self.modified_display_mode,
                       self.averaged_normalized_display_mode]:
                var.set("line")
            
            # Reset ranges
            self.hyperpol_start.set(0)
            self.hyperpol_end.set(200)
            self.depol_start.set(0)
            self.depol_end.set(200)
            
            # Enable analyze button
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error resetting tab: {str(e)}")
            raise

    def run_starting_point_simulation(self):
        """Run starting point simulation to find optimal value."""
        try:
            # Check if we have data available
            app = self.parent.master
            
            # Debug logging
            app_logger.info(f"Starting point simulation requested")
            app_logger.info(f"Checking data availability:")
            app_logger.info(f"  filtered_data: {app.filtered_data is not None}")
            app_logger.info(f"  time_data: {app.time_data is not None}")
            if app.filtered_data is not None:
                app_logger.info(f"  filtered_data length: {len(app.filtered_data)}")
            if app.time_data is not None:
                app_logger.info(f"  time_data length: {len(app.time_data)}")
            
            # Also check if we have the data property
            app_logger.info(f"  hasattr filtered_data: {hasattr(app, 'filtered_data')}")
            app_logger.info(f"  hasattr time_data: {hasattr(app, 'time_data')}")
            app_logger.info(f"  app type: {type(app)}")
            app_logger.info(f"  app dir: {[attr for attr in dir(app) if 'data' in attr.lower()]}")
            
            # Get data with fallback methods
            filtered_data = None
            time_data = None
            
            # Try property access first
            if hasattr(app, 'filtered_data') and app.filtered_data is not None:
                filtered_data = app.filtered_data
                app_logger.info("Using filtered_data from property")
            elif hasattr(app, '_filtered_data') and app._filtered_data is not None:
                filtered_data = app._filtered_data
                app_logger.info("Using filtered_data from private attribute")
            elif hasattr(app, 'data') and app.data is not None:
                filtered_data = app.data
                app_logger.info("Using raw data as filtered_data")
            
            if hasattr(app, 'time_data') and app.time_data is not None:
                time_data = app.time_data
                app_logger.info("Using time_data from property")
            elif hasattr(app, '_time_data') and app._time_data is not None:
                time_data = app._time_data
                app_logger.info("Using time_data from private attribute")
            
            # Check if we have both datasets
            if filtered_data is None:
                app_logger.warning("No filtered data available for simulation")
                messagebox.showwarning("No Data", "Please load data first before running simulation.")
                return
            
            if time_data is None:
                app_logger.warning("No time data available for simulation")
                messagebox.showwarning("No Data", "Time data not available for simulation.")
                return
            
            # Disable simulation button during processing
            self.simulate_button.state(['disabled'])
            self.simulate_button.config(text="Running Simulation...")
            
            # Get current parameters
            current_params = self.get_parameters()
            
            # Create and show simulation dialog
            self._show_simulation_dialog(filtered_data, time_data, current_params)
            
        except Exception as e:
            app_logger.error(f"Error running starting point simulation: {str(e)}")
            messagebox.showerror("Simulation Error", f"Failed to run simulation: {str(e)}")
        finally:
            # Re-enable simulation button
            if hasattr(self, 'simulate_button'):
                self.simulate_button.state(['!disabled'])
                self.simulate_button.config(text="Find Optimal Point")
    
    def _show_simulation_dialog(self, data, time_data, params):
        """Show the simulation dialog window."""
        try:
            # Create simulation dialog window
            dialog = tk.Toplevel(self.parent)
            dialog.title("Starting Point Simulation")
            dialog.geometry("800x600")
            dialog.transient(self.parent)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (800 // 2)
            y = (dialog.winfo_screenheight() // 2) - (600 // 2)
            dialog.geometry(f"800x600+{x}+{y}")
            
            # Create simulation GUI
            from src.analysis.starting_point_simulator import StartingPointSimulationGUI
            sim_gui = StartingPointSimulationGUI(dialog, data, time_data, params)
            
            # Add close handler
            def on_close():
                dialog.destroy()
            
            dialog.protocol("WM_DELETE_WINDOW", on_close)
            
            # Add apply button handler
            def apply_recommendation():
                recommended_point = sim_gui.get_recommended_starting_point()
                if recommended_point is not None:
                    # Update the starting point in the tab
                    self.norm_point.set(str(recommended_point))
                    messagebox.showinfo("Applied", f"Starting point set to {recommended_point}")
                    dialog.destroy()
                else:
                    messagebox.showwarning("No Recommendation", "No recommendation available to apply")
            
            # Override the apply method to close dialog
            sim_gui.apply_recommendation = apply_recommendation
            
        except Exception as e:
            app_logger.error(f"Error showing simulation dialog: {str(e)}")
            messagebox.showerror("Dialog Error", f"Failed to show simulation dialog: {str(e)}")

    def on_auto_optimize_change(self):
        """Handle changes to auto-optimization checkbox."""
        try:
            auto_optimize = self.auto_optimize.get()
            
            # Enable/disable manual simulation button based on auto-optimization
            if hasattr(self, 'simulate_button'):
                if auto_optimize:
                    self.simulate_button.config(text="Find Optimal Point (Manual)")
                    # Show info about auto-optimization
                    app_logger.info("Auto-optimization enabled - starting point will be optimized automatically")
                else:
                    self.simulate_button.config(text="Find Optimal Point")
                    app_logger.info("Auto-optimization disabled - manual optimization available")
            
        except Exception as e:
            app_logger.error(f"Error handling auto-optimize change: {str(e)}")
    
    def _check_analysis_available(self) -> bool:
        """Check if analysis has been run and data is available."""
        try:
            # Get the processor from the main app
            app = self.parent.master
            processor = getattr(app, 'action_potential_processor', None)
            
            if not processor:
                messagebox.showwarning("No Analysis", "Please run analysis first.")
                return False
            
            # Check if we have the required curves
            if (not hasattr(processor, 'modified_hyperpol') or 
                not hasattr(processor, 'modified_depol') or
                processor.modified_hyperpol is None or
                processor.modified_depol is None):
                messagebox.showwarning("Missing Data", 
                                     "Analysis does not contain required curves. "
                                     "Please run analysis with 'Show Modified Peaks' enabled.")
                return False
            
            return True
            
        except Exception as e:
            app_logger.error(f"Error checking analysis availability: {str(e)}")
            messagebox.showerror("Error", f"Error checking analysis: {str(e)}")
            return False
    
    def _check_curve_fitting_available(self) -> bool:
        """Check if curve fitting has been performed."""
        try:
            # Check if curve fitting panel exists
            if not hasattr(self, 'curve_fitting_panel') or not self.curve_fitting_panel:
                messagebox.showwarning("No Fitting", "Curve fitting panel not available.")
                return False
            
            # Check if fitting manager exists
            fitting_manager = getattr(self.curve_fitting_panel, 'fitting_manager', None)
            if not fitting_manager:
                messagebox.showwarning("No Fitting", "Fitting manager not initialized.")
                return False
            
            # Check if any fitting has been performed
            fitting_results = fitting_manager.get_fitting_results()
            if not fitting_results:
                messagebox.showwarning("No Fitting", "No curve fitting has been performed. "
                                                  "Please perform linear or exponential fitting first.")
                return False
            
            # Check if we have at least some fitting data
            has_fitting_data = False
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in fitting_results:
                    curve_data = fitting_results[curve_type]
                    if 'linear' in curve_data or 'exponential' in curve_data:
                        has_fitting_data = True
                        break
            
            if not has_fitting_data:
                messagebox.showwarning("No Fitting", "No valid fitting data found. "
                                                  "Please perform linear or exponential fitting first.")
                return False
            
            return True
            
        except Exception as e:
            app_logger.error(f"Error checking curve fitting availability: {str(e)}")
            messagebox.showerror("Error", f"Error checking curve fitting: {str(e)}")
            return False
    
    def collect_export_data(self) -> dict:
        """Collect all data needed for Excel export."""
        try:
            from datetime import datetime
            
            # Get the processor from the main app
            app = self.parent.master
            processor = getattr(app, 'action_potential_processor', None)
            
            if not processor:
                raise ValueError("No processor available")
            
            # Get current file information
            current_file = getattr(app, 'current_file', 'Unknown')
            filename = os.path.basename(current_file) if current_file else 'Unknown'
            
            # Extract voltage from filename if possible
            voltage = 'N/A'
            if filename and '_' in filename:
                try:
                    # Pattern: 0057_0_-100.atf
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        voltage = parts[2].split('.')[0]
                except:
                    pass
            
            # Get fitting results
            fitting_results = {}
            if hasattr(self, 'curve_fitting_panel') and self.curve_fitting_panel:
                fitting_manager = getattr(self.curve_fitting_panel, 'fitting_manager', None)
                if fitting_manager:
                    fitting_results = fitting_manager.get_fitting_results()
            
            # Get integration data (placeholder - would need actual implementation)
            integration_data = {
                'hyperpol': {
                    'integral_value': 'N/A',
                    'start_index': 'N/A',
                    'end_index': 'N/A',
                    'method': 'N/A'
                },
                'depol': {
                    'integral_value': 'N/A',
                    'start_index': 'N/A',
                    'end_index': 'N/A',
                    'method': 'N/A'
                }
            }
            
            # Get capacitance data (placeholder - would need actual implementation)
            capacitance_data = {
                'linear_capacitance': 'N/A',
                'method': 'Linear Fit',
                'notes': 'Calculated from linear fitting parameters'
            }
            
            # Compile export data
            export_data = {
                'filename': filename,
                'voltage': voltage,
                'file_path': current_file,
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_type': 'Curve Fitting',
                'export_type': 'single_file',
                'fitting_results': fitting_results,
                'integration_data': integration_data,
                'capacitance_data': capacitance_data
            }
            
            app_logger.info(f"Collected export data for {filename}")
            return export_data
            
        except Exception as e:
            app_logger.error(f"Error collecting export data: {str(e)}")
            raise
    
    def on_export_to_excel_click(self):
        """Handle Export to Excel button click."""
        try:
            # Check if curve fitting has been performed (this is the main requirement)
            if not self._check_curve_fitting_available():
                return
            
            # Collect export data
            export_data = self.collect_export_data()
            
            # Create backup before export
            from src.excel_export.export_backup_manager import backup_manager
            backup_path = backup_manager.create_backup(export_data, export_data['filename'])
            if backup_path:
                app_logger.info(f"Backup created: {backup_path}")
            
            # Open file dialog for save location
            from tkinter import filedialog
            filetypes = [
                ('Excel files', '*.xlsx'),
                ('All files', '*.*')
            ]
            
            filename = export_data['filename']
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            default_filename = f"{filename}_curve_analysis.xlsx"
            
            filepath = filedialog.asksaveasfilename(
                title="Save Excel Export",
                defaultextension=".xlsx",
                filetypes=filetypes,
                initialvalue=default_filename
            )
            
            if not filepath:
                return  # User cancelled
            
            # Export to Excel
            from src.excel_export.curve_analysis_export import export_curve_analysis_to_excel
            
            success = export_curve_analysis_to_excel(export_data, filepath)
            
            if success:
                messagebox.showinfo("Export Successful", f"Data exported successfully to:\n{filepath}")
                app_logger.info(f"Excel export completed: {filepath}")
                
                # Log export action
                main_app = self._get_main_app()
                if main_app and hasattr(main_app, 'action_logger') and main_app.action_logger:
                    try:
                        main_app.action_logger.log_export('excel_single', filepath)
                    except Exception as e:
                        app_logger.warning(f"Failed to log export: {e}")
            else:
                messagebox.showerror("Export Failed", "Failed to export data to Excel. Check logs for details.")
                app_logger.error("Excel export failed")
                
        except Exception as e:
            app_logger.error(f"Error in Excel export: {str(e)}")
            messagebox.showerror("Export Error", f"An error occurred during export:\n{str(e)}")
    
    def on_export_sets_to_excel_click(self):
        """Handle Export Sets to Excel button click."""
        try:
            # Check if curve fitting has been performed (this is the main requirement)
            if not self._check_curve_fitting_available():
                return
            
            # Use the set-based export functionality
            from src.excel_export.set_based_export import export_sets_to_excel
            
            main_app = self._get_main_app()
            if main_app:
                success = export_sets_to_excel(main_app)
            
            if success:
                app_logger.info("Set-based Excel export completed")
            else:
                app_logger.warning("Set-based Excel export failed or was cancelled")
                
        except Exception as e:
            app_logger.error(f"Error in set-based Excel export: {str(e)}")
            messagebox.showerror("Export Error", f"An error occurred during set-based export:\n{str(e)}")
    
    def _get_main_app(self):
        """Get reference to main application."""
        try:
            # Try to get main app from callback
            if hasattr(self.update_callback, '__self__'):
                return self.update_callback.__self__
            # Try parent hierarchy
            widget = self.parent
            while widget:
                if hasattr(widget, 'action_logger'):
                    return widget
                widget = getattr(widget, 'master', None)
            return None
        except Exception as e:
            app_logger.warning(f"Could not get main app reference: {e}")
            return None
    
    def start_fast_track(self):
        """Start fast-track automated workflow."""
        try:
            main_app = self._get_main_app()
            if not main_app or not hasattr(main_app, 'fast_track_mode'):
                messagebox.showerror("Error", "Fast-track mode not available")
                return
            
            # Confirm with user
            result = messagebox.askyesno(
                "Fast Track Mode",
                "This will automate the analysis workflow:\n\n"
                "1. Set filters (Savitzky-Golay, Butterworth)\n"
                "2. Run analysis\n"
                "3. Perform fittings (if points are saved)\n"
                "4. Export to Excel\n\n"
                "Continue?"
            )
            
            if not result:
                return
            
            # Disable button during execution
            self.fast_track_button.state(['disabled'])
            self.status_text.set("Fast-track mode running...")
            
            # Execute workflow in background
            def execute():
                try:
                    success = main_app.fast_track_mode.execute_workflow()
                    self.parent.after(0, lambda: self._fast_track_complete(success))
                except Exception as e:
                    app_logger.error(f"Fast-track error: {e}")
                    self.parent.after(0, lambda: self._fast_track_complete(False, str(e)))
            
            import threading
            thread = threading.Thread(target=execute, daemon=True)
            thread.start()
            
        except Exception as e:
            app_logger.error(f"Error starting fast-track: {e}")
            messagebox.showerror("Error", f"Failed to start fast-track mode: {e}")
            self.fast_track_button.state(['!disabled'])
    
    def _fast_track_complete(self, success: bool, error_msg: str = None):
        """Handle fast-track completion."""
        self.fast_track_button.state(['!disabled'])
        if success:
            self.status_text.set("Fast-track completed successfully")
            messagebox.showinfo("Success", "Fast-track workflow completed successfully!")
        else:
            self.status_text.set(f"Fast-track failed: {error_msg or 'Unknown error'}")
            messagebox.showerror("Error", f"Fast-track workflow failed:\n{error_msg or 'Unknown error'}")
    
    def save_session_log(self):
        """Save current session log to file."""
        try:
            main_app = self._get_main_app()
            if not main_app or not hasattr(main_app, 'action_logger'):
                messagebox.showwarning("Warning", "Action logger not available")
                return
            
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                title="Save Session Log",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                filepath = main_app.action_logger.save_session(filename)
                messagebox.showinfo("Success", f"Session log saved to:\n{filepath}")
                app_logger.info(f"Session log saved: {filepath}")
        
        except Exception as e:
            app_logger.error(f"Error saving session log: {e}")
            messagebox.showerror("Error", f"Failed to save session log: {e}")