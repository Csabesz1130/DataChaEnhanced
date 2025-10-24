"""
Curve Fitting GUI Module for DataChaEnhanced
============================================
Location: src/gui/curve_fitting_gui.py

This module adds curve fitting controls to the existing GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import Optional
import logging
import csv

logger = logging.getLogger(__name__)

class CurveFittingPanel:
    """Panel for curve fitting controls integrated into ActionPotentialTab."""
    
    def __init__(self, parent_frame, main_app):
        """Initialize the curve fitting panel."""
        self.parent_frame = parent_frame
        self.main_app = main_app
        self.fitting_manager = None
        
        # Status variables
        self.status_var = tk.StringVar()
        self.status_var.set("Ready for curve fitting")
        
        # Result variables
        self.result_vars = {}
        for curve_type in ['hyperpol', 'depol']:
            self.result_vars[curve_type] = {
                'linear_equation': tk.StringVar(),
                'linear_slope': tk.StringVar(),
                'linear_intercept': tk.StringVar(),
                'linear_r2': tk.StringVar(),
                'exp_equation': tk.StringVar(),
                'exp_A': tk.StringVar(),
                'exp_tau': tk.StringVar(),
                'exp_C': tk.StringVar(),
                'exp_r2': tk.StringVar()
            }
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all GUI widgets for curve fitting."""
        # Main frame with border
        main_frame = ttk.LabelFrame(self.parent_frame, text="📊 Manual Curve Fitting", padding=10)
        main_frame.pack(fill='both', expand=False, pady=(10, 0))
        
        # Instructions
        instructions_frame = ttk.Frame(main_frame)
        instructions_frame.pack(fill='x', pady=(0, 10))
        
        instructions_text = """1. Válasszon görbe típust (Hyperpol/Depol)
2. Kattintson 'Start Linear Fit' → válasszon 2 pontot az egyenes illesztéshez
3. Kattintson 'Start Exp Fit' → válasszon 1 pontot az exponenciális kezdetéhez
4. Az eredmények automatikusan megjelennek"""
        
        ttk.Label(instructions_frame, text=instructions_text, 
                 font=('Arial', 9), justify='left').pack(anchor='w')
        
        # Control panels in horizontal layout
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Hyperpolarization controls
        self._create_curve_controls(controls_frame, 'hyperpol', 'Hyperpolarization', 0)
        
        # Separator
        ttk.Separator(controls_frame, orient='vertical').grid(row=0, column=1, rowspan=5, 
                                                              sticky='ns', padx=10)
        
        # Depolarization controls
        self._create_curve_controls(controls_frame, 'depol', 'Depolarization', 2)
        
        # Global controls
        global_frame = ttk.Frame(main_frame)
        global_frame.pack(fill='x', pady=(10, 5))
        
        ttk.Button(global_frame, text="🗑️ Clear All", 
                  command=lambda: self.clear_fits(None)).pack(side='left', padx=2)
        ttk.Button(global_frame, text="💾 Export Results", 
                  command=self.export_results).pack(side='left', padx=2)
        ttk.Button(global_frame, text="📊 Export to Excel (Single)", 
                  command=self.export_to_excel_single).pack(side='left', padx=2)
        ttk.Button(global_frame, text="📊 Export Sets to Excel", 
                  command=self.export_to_excel_sets).pack(side='left', padx=2)
        # Apply corrections buttons - separate for each curve type
        ttk.Button(global_frame, text="📈 Apply Hyperpol", 
                  command=lambda: self.apply_corrections('hyperpol')).pack(side='left', padx=2)
        ttk.Button(global_frame, text="📈 Apply Depol", 
                  command=lambda: self.apply_corrections('depol')).pack(side='left', padx=2)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', pady=(5, 0))
        ttk.Label(status_frame, textvariable=self.status_var,
                 relief='sunken', anchor='w').pack(fill='x')
        
        # Results display (compact)
        self._create_results_display(main_frame)
    
    def _create_curve_controls(self, parent, curve_type, label, column):
        """Create control buttons for a specific curve type."""
        frame = ttk.LabelFrame(parent, text=label, padding=5)
        frame.grid(row=0, column=column, sticky='ew', padx=5)
        
        # Store button references
        setattr(self, f'{curve_type}_linear_btn',
                ttk.Button(frame, text="📍 Start Linear Fit",
                          command=lambda: self.start_linear_fitting(curve_type)))
        getattr(self, f'{curve_type}_linear_btn').pack(fill='x', pady=2)
        
        setattr(self, f'{curve_type}_exp_btn',
                ttk.Button(frame, text="📈 Start Exp Fit",
                          command=lambda: self.start_exponential_fitting(curve_type)))
        getattr(self, f'{curve_type}_exp_btn').pack(fill='x', pady=2)
        
        ttk.Button(frame, text="Clear", 
                  command=lambda: self.clear_fits(curve_type)).pack(fill='x', pady=2)
        
        # Add Reset to Original button
        setattr(self, f'{curve_type}_reset_original_btn',
                ttk.Button(frame, text="🔄 Reset to Original",
                          command=lambda: self.reset_to_original(curve_type)))
        getattr(self, f'{curve_type}_reset_original_btn').pack(fill='x', pady=2)
        
        # Add Integration Range Selection button
        setattr(self, f'{curve_type}_integration_btn',
                ttk.Button(frame, text="📏 Select Integration Range",
                          command=lambda: self.start_integration_selection(curve_type)))
        getattr(self, f'{curve_type}_integration_btn').pack(fill='x', pady=2)
    
    def _create_results_display(self, parent):
        """Create compact results display."""
        results_frame = ttk.LabelFrame(parent, text="Fitting Results", padding=5)
        results_frame.pack(fill='x', pady=(5, 0))
        
        # Create notebook for results
        notebook = ttk.Notebook(results_frame, height=150)
        notebook.pack(fill='x')
        
        # Hyperpol results
        hyperpol_frame = ttk.Frame(notebook)
        notebook.add(hyperpol_frame, text="Hyperpol")
        self._create_result_widgets(hyperpol_frame, 'hyperpol')
        
        # Depol results
        depol_frame = ttk.Frame(notebook)
        notebook.add(depol_frame, text="Depol")
        self._create_result_widgets(depol_frame, 'depol')
    
    def _create_result_widgets(self, parent, curve_type):
        """Create result display widgets for a specific curve."""
        # Two columns: Linear and Exponential
        linear_frame = ttk.Frame(parent)
        linear_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        ttk.Label(linear_frame, text="Linear Fit:", font=('Arial', 9, 'bold')).pack(anchor='w')
        ttk.Label(linear_frame, textvariable=self.result_vars[curve_type]['linear_equation'],
                 font=('Courier', 8)).pack(anchor='w')
        ttk.Label(linear_frame, textvariable=self.result_vars[curve_type]['linear_r2'],
                 font=('Arial', 8)).pack(anchor='w')
        
        exp_frame = ttk.Frame(parent)
        exp_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        ttk.Label(exp_frame, text="Exponential Fit:", font=('Arial', 9, 'bold')).pack(anchor='w')
        ttk.Label(exp_frame, textvariable=self.result_vars[curve_type]['exp_equation'],
                 font=('Courier', 8)).pack(anchor='w')
        ttk.Label(exp_frame, textvariable=self.result_vars[curve_type]['exp_r2'],
                 font=('Arial', 8)).pack(anchor='w')
    
    def initialize_fitting_manager(self, figure, ax):
        """Initialize the fitting manager with plot references."""
        from src.analysis.curve_fitting_manager import CurveFittingManager
        
        self.fitting_manager = CurveFittingManager(figure, ax)
        self.fitting_manager.on_fit_complete = self.on_fitting_completed
        
        # Add reference to main app for plot refresh functionality
        if hasattr(self, 'main_app'):
            self.fitting_manager.main_app = self.main_app
        
        self.update_curve_data()
        logger.info("Fitting manager initialized")
    
    def update_curve_data(self):
        """Update curve data in the fitting manager."""
        if not self.fitting_manager:
            return
        
        processor = getattr(self.main_app, 'action_potential_processor', None)
        if not processor:
            return
        
        # Store original data if not already stored
        self._store_original_data(processor)
        
        # Update hyperpol data
        if (hasattr(processor, 'modified_hyperpol') and 
            processor.modified_hyperpol is not None):
            self.fitting_manager.set_curve_data(
                'hyperpol',
                processor.modified_hyperpol,
                processor.modified_hyperpol_times
            )
        
        # Update depol data
        if (hasattr(processor, 'modified_depol') and 
            processor.modified_depol is not None):
            self.fitting_manager.set_curve_data(
                'depol',
                processor.modified_depol,
                processor.modified_depol_times
            )
    
    def _store_original_data(self, processor):
        """Store original curve data for reset functionality."""
        try:
            # Store hyperpol original data if not already stored
            if (hasattr(processor, 'modified_hyperpol') and hasattr(processor, 'modified_hyperpol_times') and
                processor.modified_hyperpol is not None and
                (not hasattr(processor, 'original_hyperpol') or processor.original_hyperpol is None)):
                processor.original_hyperpol = processor.modified_hyperpol.copy()
                processor.original_hyperpol_times = processor.modified_hyperpol_times.copy()
                logger.info("Stored original hyperpol data")
            
            # Store depol original data if not already stored
            if (hasattr(processor, 'modified_depol') and hasattr(processor, 'modified_depol_times') and
                processor.modified_depol is not None and
                (not hasattr(processor, 'original_depol') or processor.original_depol is None)):
                processor.original_depol = processor.modified_depol.copy()
                processor.original_depol_times = processor.modified_depol_times.copy()
                logger.info("Stored original depol data")
                
        except Exception as e:
            logger.error(f"Failed to store original data: {str(e)}")
    
    def start_linear_fitting(self, curve_type: str):
        """Start linear fitting for the specified curve."""
        if not self.fitting_manager:
            messagebox.showerror("Error", "Fitting manager not initialized")
            return
        
        self.update_curve_data()
        
        if self.fitting_manager.curve_data[curve_type]['data'] is None:
            messagebox.showerror("Error", f"No {curve_type} data available")
            return
        
        self.fitting_manager.start_linear_selection(curve_type)
        self.status_var.set(f"Click 2 points on {curve_type} curve for linear fit")
        self._disable_buttons_except(f'{curve_type}_linear_btn')
    
    def start_exponential_fitting(self, curve_type: str):
        """Start exponential fitting for the specified curve."""
        if not self.fitting_manager:
            messagebox.showerror("Error", "Fitting manager not initialized")
            return
        
        # No longer require linear fit for exponential fitting
        
        self.fitting_manager.start_exp_selection(curve_type)
        self.status_var.set(f"Click 2 points on {curve_type} curve for exponential fit")
        self._disable_buttons_except(f'{curve_type}_exp_btn')
    
    def start_integration_selection(self, curve_type: str):
        """Start integration range selection for the specified curve."""
        if not self.fitting_manager:
            messagebox.showerror("Error", "Fitting manager not initialized")
            return
        
        self.fitting_manager.start_integration_selection(curve_type)
        self.status_var.set(f"Click 2 points on {curve_type} curve to define integration range")
        self._disable_buttons_except(f'{curve_type}_integration_btn')
    
    def clear_fits(self, curve_type: Optional[str]):
        """Clear fitting results."""
        if self.fitting_manager:
            self.fitting_manager.clear_fits(curve_type)
        self._update_results_display()
        self.status_var.set("Fits cleared")
        self._enable_all_buttons()
    
    def reset_to_original(self, curve_type: str):
        """Reset the curve to its original state before any corrections."""
        try:
            if not self.fitting_manager:
                messagebox.showwarning("Warning", "No fitting manager available")
                return
            
            # Get the processor
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if not processor:
                messagebox.showwarning("Warning", "No processor data available")
                return
            
            # Check if we have original data stored
            original_data_attr = f'original_{curve_type}'
            original_times_attr = f'original_{curve_type}_times'
            
            if not (hasattr(processor, original_data_attr) and hasattr(processor, original_times_attr)):
                messagebox.showwarning("Warning", f"No original {curve_type} data available. Please run analysis first.")
                return
            
            # Reset the modified curve to original
            if curve_type == 'hyperpol':
                logger.info(f"Before reset - modified_hyperpol: {processor.modified_hyperpol}")
                logger.info(f"Before reset - original_hyperpol: {processor.original_hyperpol}")
                processor.modified_hyperpol = processor.original_hyperpol.copy()
                processor.modified_hyperpol_times = processor.original_hyperpol_times.copy()
                logger.info(f"After reset - modified_hyperpol: {processor.modified_hyperpol}")
                logger.info("Reset hyperpol curve to original")
            elif curve_type == 'depol':
                logger.info(f"Before reset - modified_depol: {processor.modified_depol}")
                logger.info(f"Before reset - original_depol: {processor.original_depol}")
                processor.modified_depol = processor.original_depol.copy()
                processor.modified_depol_times = processor.original_depol_times.copy()
                logger.info(f"After reset - modified_depol: {processor.modified_depol}")
                logger.info("Reset depol curve to original")
            
            # Update the curve data in the fitting manager with the reset data
            if curve_type == 'hyperpol':
                self.fitting_manager.curve_data['hyperpol']['data'] = processor.modified_hyperpol.copy()
                self.fitting_manager.curve_data['hyperpol']['times'] = processor.modified_hyperpol_times.copy()
                logger.info("Updated fitting manager curve data for hyperpol with reset values")
            elif curve_type == 'depol':
                self.fitting_manager.curve_data['depol']['data'] = processor.modified_depol.copy()
                self.fitting_manager.curve_data['depol']['times'] = processor.modified_depol_times.copy()
                logger.info("Updated fitting manager curve data for depol with reset values")
            
            # Reload the plot with preserved zoom state
            if hasattr(self.main_app, 'update_plot_with_processed_data'):
                try:
                    # Explicitly preserve zoom by setting force_full_range=False and force_auto_scale=False
                    self.main_app.update_plot_with_processed_data(
                        getattr(processor, 'processed_data', None),
                        getattr(processor, 'orange_curve', None),
                        getattr(processor, 'orange_times', None),
                        getattr(processor, 'normalized_curve', None),
                        getattr(processor, 'normalized_curve_times', None),
                        getattr(processor, 'average_curve', None),
                        getattr(processor, 'average_curve_times', None),
                        force_full_range=False,
                        force_auto_scale=False
                    )
                    logger.info(f"Plot refreshed after {curve_type} reset to original with preserved zoom")
                except Exception as e:
                    logger.error(f"Failed to refresh plot after reset: {str(e)}")
                    # Try alternative plot update method
                    if hasattr(self.main_app, 'update_plot'):
                        self.main_app.update_plot()
            
            # Update status
            self.status_var.set(f"{curve_type.title()} curve reset to original")
            messagebox.showinfo("Success", f"{curve_type.title()} curve has been reset to its original state")
            
        except Exception as e:
            logger.error(f"Failed to reset {curve_type} to original: {str(e)}")
            messagebox.showerror("Error", f"Failed to reset {curve_type} to original: {str(e)}")
    
    def apply_corrections(self, curve_type=None):
        """Apply linear corrections to curves."""
        if not self.fitting_manager:
            return
        
        corrections_applied = []
        
        # If curve_type is specified, only apply to that curve
        curves_to_process = [curve_type] if curve_type else ['hyperpol', 'depol']
        
        for curve in curves_to_process:
            if self.fitting_manager.fitted_curves[curve]['linear_params']:
                # Both curves should subtract the linear trend to remove drift
                operation = 'subtract'
                result = self.fitting_manager.apply_linear_correction(curve, operation)
                if result:
                    corrections_applied.append(f"{curve}: {operation}")
                    
                    # Update the processor with corrected data
                    processor = getattr(self.main_app, 'action_potential_processor', None)
                    if processor:
                        if curve == 'hyperpol':
                            processor.corrected_hyperpol = result['corrected']
                        else:
                            processor.corrected_depol = result['corrected']
        
        if corrections_applied:
            # Reload the plot with preserved zoom state after applying corrections
            if hasattr(self.main_app, 'update_plot_with_processed_data'):
                try:
                    processor = getattr(self.main_app, 'action_potential_processor', None)
                    if processor:
                        self.main_app.update_plot_with_processed_data(
                            getattr(processor, 'processed_data', None),
                            getattr(processor, 'orange_curve', None),
                            getattr(processor, 'orange_times', None),
                            getattr(processor, 'normalized_curve', None),
                            getattr(processor, 'normalized_curve_times', None),
                            getattr(processor, 'average_curve', None),
                            getattr(processor, 'average_curve_times', None),
                            force_full_range=False,
                            force_auto_scale=False
                        )
                        logger.info("Plot refreshed after applying corrections with preserved zoom")
                except Exception as e:
                    logger.error(f"Failed to refresh plot after corrections: {str(e)}")
            
            messagebox.showinfo("Success", f"Corrections applied:\n" + "\n".join(corrections_applied))
            if curve_type:
                self.status_var.set(f"Linear correction applied to {curve_type}")
            else:
                self.status_var.set("Linear corrections applied")
        else:
            messagebox.showwarning("Warning", "No linear fits available for correction")
    
    def export_results(self):
        """Export fitting results to CSV file."""
        if not self.fitting_manager:
            return
        
        results = self.fitting_manager.get_fitting_results()
        if not results:
            messagebox.showwarning("Warning", "No fitting results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Fitting Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(["CURVE FITTING RESULTS"])
                    writer.writerow([])
                    
                    for curve_type in ['hyperpol', 'depol']:
                        if curve_type in results:
                            writer.writerow([f"{curve_type.upper()} CURVE:"])
                            
                            if 'linear' in results[curve_type]:
                                linear = results[curve_type]['linear']
                                writer.writerow(["Linear Fit:"])
                                writer.writerow(["Equation:", linear['equation']])
                                writer.writerow(["Slope:", f"{linear['slope']:.6f}"])
                                writer.writerow(["Intercept:", f"{linear['intercept']:.6f}"])
                                writer.writerow(["R²:", f"{linear['r_squared']:.6f}"])
                                writer.writerow([])
                            
                            if 'exponential' in results[curve_type]:
                                exp = results[curve_type]['exponential']
                                writer.writerow(["Exponential Fit:"])
                                writer.writerow(["Equation:", exp['equation']])
                                writer.writerow(["Amplitude (A):", f"{exp['A']:.6f}"])
                                writer.writerow(["Time constant (τ):", f"{exp['tau']:.6f}"])
                                writer.writerow(["Baseline (C):", f"{exp['C']:.6f}"])
                                writer.writerow(["R²:", f"{exp['r_squared']:.6f}"])
                                writer.writerow([])
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def on_fitting_completed(self, curve_type: str, fit_type: str):
        """Callback when fitting is completed."""
        self._update_results_display()
        self._enable_all_buttons()
        self.status_var.set(f"{fit_type.title()} fitting completed for {curve_type}")
    
    def _update_results_display(self):
        """Update the results display with current fitting results."""
        if not self.fitting_manager:
            return
        
        results = self.fitting_manager.get_fitting_results()
        
        for curve_type in ['hyperpol', 'depol']:
            # Clear previous results
            for var in self.result_vars[curve_type].values():
                var.set("")
            
            if curve_type in results:
                # Linear results
                if 'linear' in results[curve_type]:
                    linear = results[curve_type]['linear']
                    self.result_vars[curve_type]['linear_equation'].set(linear['equation'])
                    self.result_vars[curve_type]['linear_r2'].set(f"R² = {linear['r_squared']:.4f}")
                
                # Exponential results
                if 'exponential' in results[curve_type]:
                    exp = results[curve_type]['exponential']
                    # Shorten equation for display
                    if len(exp['equation']) > 40:
                        eq_short = f"A={exp['A']:.2f}, τ={exp['tau']:.3f}, C={exp['C']:.2f}"
                    else:
                        eq_short = exp['equation']
                    self.result_vars[curve_type]['exp_equation'].set(eq_short)
                    self.result_vars[curve_type]['exp_r2'].set(f"R² = {exp['r_squared']:.4f}")
    
    def _disable_buttons_except(self, except_button: str):
        """Disable all buttons except the specified one."""
        buttons = ['hyperpol_linear_btn', 'hyperpol_exp_btn', 'hyperpol_reset_original_btn', 'hyperpol_integration_btn',
                  'depol_linear_btn', 'depol_exp_btn', 'depol_reset_original_btn', 'depol_integration_btn']
        
        for btn_name in buttons:
            if btn_name != except_button:
                btn = getattr(self, btn_name, None)
                if btn:
                    btn.config(state='disabled')
    
    def _enable_all_buttons(self):
        """Enable all buttons."""
        buttons = ['hyperpol_linear_btn', 'hyperpol_exp_btn', 'hyperpol_reset_original_btn', 'hyperpol_integration_btn',
                  'depol_linear_btn', 'depol_exp_btn', 'depol_reset_original_btn', 'depol_integration_btn']
        
        for btn_name in buttons:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.config(state='normal')
    
    def export_to_excel_single(self):
        """Export current file to Excel (single file export)."""
        try:
            # Get the action potential tab from the main app
            app = self.main_app
            if hasattr(app, 'action_potential_tab'):
                app.action_potential_tab.on_export_to_excel_click()
            else:
                messagebox.showerror("Error", "Action potential tab not available.")
        except Exception as e:
            logger.error(f"Error in single Excel export: {str(e)}")
            messagebox.showerror("Export Error", f"An error occurred during single file export:\n{str(e)}")
    
    def export_to_excel_sets(self):
        """Export multiple files organized by sets to Excel."""
        try:
            # Get the action potential tab from the main app
            app = self.main_app
            if hasattr(app, 'action_potential_tab'):
                app.action_potential_tab.on_export_sets_to_excel_click()
            else:
                messagebox.showerror("Error", "Action potential tab not available.")
        except Exception as e:
            logger.error(f"Error in sets Excel export: {str(e)}")
            messagebox.showerror("Export Error", f"An error occurred during sets export:\n{str(e)}")