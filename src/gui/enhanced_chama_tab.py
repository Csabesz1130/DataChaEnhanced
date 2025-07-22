# src/gui/enhanced_chama_tab.py
"""
Enhanced ChaMa Tab - Integration of ChaMa VB features into Python Signal Analyzer

This tab provides the core ChaMa functionality including:
- Negative control processing
- Ohmic component removal
- Charge movement calculation
- Protocol parameter management
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pickle
import os
from typing import Optional, Dict, Any

from src.analysis.negative_control_processor import NegativeControlProcessor, ProtocolParameters
from src.utils.logger import app_logger


class ChamaTab:
    """Enhanced ChaMa functionality tab"""
    
    def __init__(self, parent, app_reference):
        """Initialize the ChaMa tab with enhanced functionality"""
        self.parent = parent
        self.app = app_reference
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Initialize processor
        self.processor = NegativeControlProcessor()
        
        # Storage for current data
        self.current_data = None
        self.current_time = None
        self.processed_data = None
        
        # UI state variables
        self.correction_mode = tk.StringVar(value="trace")
        self.auto_baseline = tk.BooleanVar(value=True)
        self.show_fit = tk.BooleanVar(value=True)
        
        # Setup UI
        self.setup_ui()
        
        app_logger.info("Enhanced ChaMa tab initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for different sections
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Protocol Parameters Tab
        self.setup_protocol_tab()
        
        # Negative Control Tab
        self.setup_negative_control_tab()
        
        # Charge Movement Tab
        self.setup_charge_movement_tab()
        
        # Analysis Results Tab
        self.setup_results_tab()
        
    def setup_protocol_tab(self):
        """Setup protocol parameters configuration"""
        protocol_frame = ttk.Frame(self.notebook)
        self.notebook.add(protocol_frame, text="Protocol Setup")
        
        # Main container with scrollbar
        canvas = tk.Canvas(protocol_frame)
        scrollbar = ttk.Scrollbar(protocol_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Protocol parameter entries
        self.param_vars = {}
        
        # Timing parameters
        timing_frame = ttk.LabelFrame(scrollable_frame, text="Timing Parameters", padding=10)
        timing_frame.pack(fill='x', padx=5, pady=5)
        
        timing_params = [
            ("baseline_duration", "Baseline Duration (ms)", 10.0),
            ("neg_control_on_duration", "Negative Control ON (ms)", 25.0),
            ("neg_control_off_duration", "Negative Control OFF (ms)", 25.0),
            ("test_on_duration", "Test Pulse ON (ms)", 20.0),
            ("test_off_duration", "Test Pulse OFF (ms)", 20.0),
            ("sampling_interval", "Sampling Interval (ms)", 0.1)
        ]
        
        for i, (param, label, default) in enumerate(timing_params):
            ttk.Label(timing_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(timing_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.param_vars[param] = var
        
        # Voltage parameters
        voltage_frame = ttk.LabelFrame(scrollable_frame, text="Voltage Parameters", padding=10)
        voltage_frame.pack(fill='x', padx=5, pady=5)
        
        voltage_params = [
            ("neg_control_v1", "Negative Control V1 (mV)", -80.0),
            ("neg_control_v2", "Negative Control V2 (mV)", 0.0),
            ("test_pulse_v1", "Test Pulse V1 (mV)", -80.0),
            ("test_pulse_v2", "Test Pulse V2 (mV)", 0.0),
            ("test_pulse_v3", "Test Pulse V3 (mV)", -80.0)
        ]
        
        for i, (param, label, default) in enumerate(voltage_params):
            ttk.Label(voltage_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(voltage_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.param_vars[param] = var
        
        # Acquisition parameters
        acq_frame = ttk.LabelFrame(scrollable_frame, text="Acquisition Parameters", padding=10)
        acq_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(acq_frame, text="Number of Control Traces").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.param_vars["num_control_traces"] = tk.IntVar(value=3)
        ttk.Entry(acq_frame, textvariable=self.param_vars["num_control_traces"], width=15).grid(row=0, column=1, padx=5, pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(button_frame, text="Load Protocol", command=self.load_protocol).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Protocol", command=self.save_protocol).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_protocol).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Apply Parameters", command=self.apply_parameters).pack(side='left', padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_negative_control_tab(self):
        """Setup negative control processing"""
        neg_control_frame = ttk.Frame(self.notebook)
        self.notebook.add(neg_control_frame, text="Negative Control")
        
        # Control panel
        control_panel = ttk.LabelFrame(neg_control_frame, text="Control Operations", padding=10)
        control_panel.pack(fill='x', padx=5, pady=5)
        
        # Auto baseline checkbox
        ttk.Checkbutton(control_panel, text="Automatic Baseline Subtraction", 
                       variable=self.auto_baseline).pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Calculate Negative Control", 
                  command=self.calculate_negative_control).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Remove Ohmic Component", 
                  command=self.remove_ohmic_component).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Stored Control", 
                  command=self.load_stored_control).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Control", 
                  command=self.save_negative_control).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(neg_control_frame, text="Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.neg_control_fig = Figure(figsize=(10, 6), dpi=80)
        self.neg_control_ax = self.neg_control_fig.add_subplot(111)
        self.neg_control_canvas = FigureCanvasTkAgg(self.neg_control_fig, results_frame)
        self.neg_control_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Status
        self.neg_control_status = tk.StringVar(value="Ready to process negative control")
        ttk.Label(results_frame, textvariable=self.neg_control_status).pack(pady=5)
        
    def setup_charge_movement_tab(self):
        """Setup charge movement calculation"""
        charge_frame = ttk.Frame(self.notebook)
        self.notebook.add(charge_frame, text="Charge Movement")
        
        # Control panel
        control_panel = ttk.LabelFrame(charge_frame, text="Charge Movement Options", padding=10)
        control_panel.pack(fill='x', padx=5, pady=5)
        
        # Correction mode selection
        ttk.Label(control_panel, text="Correction Mode:").pack(anchor='w')
        mode_frame = ttk.Frame(control_panel)
        mode_frame.pack(fill='x', pady=5)
        
        modes = [("trace", "Trace Mode"), ("on", "ON Mode"), ("off", "OFF Mode"), ("average", "Average Mode")]
        for value, text in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.correction_mode, 
                           value=value).pack(side='left', padx=10)
        
        # Options
        ttk.Checkbutton(control_panel, text="Show Linear Fit", variable=self.show_fit).pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Calculate Charge Movement", 
                  command=self.calculate_charge_movement).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_charge_movement).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Plot in Main Window", 
                  command=self.plot_in_main).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(charge_frame, text="Charge Movement Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.charge_fig = Figure(figsize=(10, 6), dpi=80)
        self.charge_ax = self.charge_fig.add_subplot(111)
        self.charge_canvas = FigureCanvasTkAgg(self.charge_fig, results_frame)
        self.charge_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Status
        self.charge_status = tk.StringVar(value="Negative control required before charge movement calculation")
        ttk.Label(results_frame, textvariable=self.charge_status).pack(pady=5)
        
    def setup_results_tab(self):
        """Setup analysis results and summary"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Analysis Summary")
        
        # Summary text widget
        summary_frame = ttk.LabelFrame(results_frame, text="Processing Summary", padding=10)
        summary_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, wrap='word', height=20, width=80)
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        # Control buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Update Summary", command=self.update_summary).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Summary", command=self.export_summary).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
    
    def apply_parameters(self):
        """Apply protocol parameters to processor"""
        try:
            params = {}
            for param_name, var in self.param_vars.items():
                params[param_name] = var.get()
            
            self.processor.set_protocol_parameters(params)
            messagebox.showinfo("Success", "Protocol parameters applied successfully")
            app_logger.info("Protocol parameters applied")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply parameters: {str(e)}")
            app_logger.error(f"Error applying parameters: {str(e)}")
    
    def calculate_negative_control(self):
        """Calculate negative control from current data"""
        try:
            if self.app.data is None:
                messagebox.showwarning("Warning", "No data loaded")
                return
            
            # Get current data
            self.current_data = self.app.data.copy()
            self.current_time = self.app.time_data.copy() if self.app.time_data is not None else None
            
            # Apply current parameters
            self.apply_parameters()
            
            # Calculate negative control
            neg_control = self.processor.average_control_pulses(self.current_data, self.current_time)
            
            # Plot results
            self.plot_negative_control(neg_control)
            
            self.neg_control_status.set(f"Negative control calculated: {len(neg_control)} points")
            app_logger.info("Negative control calculated successfully")
            
        except Exception as e:
            error_msg = f"Error calculating negative control: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.neg_control_status.set(error_msg)
            app_logger.error(error_msg)
    
    def plot_negative_control(self, neg_control):
        """Plot negative control results"""
        self.neg_control_ax.clear()
        
        if neg_control is not None:
            time_axis = np.arange(len(neg_control)) * self.param_vars["sampling_interval"].get()
            self.neg_control_ax.plot(time_axis, neg_control, 'b-', linewidth=2, label='Negative Control')
            
            self.neg_control_ax.set_xlabel('Time (ms)')
            self.neg_control_ax.set_ylabel('Current (nA)')
            self.neg_control_ax.set_title('Averaged Negative Control')
            self.neg_control_ax.grid(True, alpha=0.3)
            self.neg_control_ax.legend()
        
        self.neg_control_canvas.draw()
    
    def remove_ohmic_component(self):
        """Remove ohmic component from negative control"""
        try:
            if self.processor.negative_control is None:
                messagebox.showwarning("Warning", "Calculate negative control first")
                return
            
            corrected_control = self.processor.remove_ohmic_component(self.processor.negative_control)
            
            # Update plot with both original and corrected
            self.neg_control_ax.clear()
            
            time_axis = np.arange(len(corrected_control)) * self.param_vars["sampling_interval"].get()
            self.neg_control_ax.plot(time_axis, self.processor.negative_control, 'b-', 
                                   alpha=0.5, label='Original')
            self.neg_control_ax.plot(time_axis, corrected_control, 'r-', 
                                   linewidth=2, label='Ohmic Corrected')
            
            if self.show_fit.get() and self.processor.ohmic_fit_params is not None:
                x_fit = np.arange(len(corrected_control))
                ohmic_fit = np.polyval(self.processor.ohmic_fit_params, x_fit)
                self.neg_control_ax.plot(time_axis, ohmic_fit, 'g--', 
                                       label='Linear Fit')
            
            self.neg_control_ax.set_xlabel('Time (ms)')
            self.neg_control_ax.set_ylabel('Current (nA)')
            self.neg_control_ax.set_title('Negative Control - Ohmic Component Removed')
            self.neg_control_ax.grid(True, alpha=0.3)
            self.neg_control_ax.legend()
            self.neg_control_canvas.draw()
            
            # Update negative control in processor
            self.processor.negative_control = corrected_control
            
            self.neg_control_status.set("Ohmic component removed successfully")
            app_logger.info("Ohmic component removed from negative control")
            
        except Exception as e:
            error_msg = f"Error removing ohmic component: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.neg_control_status.set(error_msg)
            app_logger.error(error_msg)
    
    def calculate_charge_movement(self):
        """Calculate charge movement using current correction mode"""
        try:
            if self.processor.negative_control is None:
                messagebox.showwarning("Warning", "Calculate negative control first")
                return
            
            if self.current_data is None:
                messagebox.showwarning("Warning", "No current trace available")
                return
            
            mode = self.correction_mode.get()
            charge_movement = self.processor.calculate_charge_movement(self.current_data, mode=mode)
            
            # Plot results
            self.plot_charge_movement(charge_movement)
            
            self.charge_status.set(f"Charge movement calculated using {mode} mode")
            app_logger.info(f"Charge movement calculated using {mode} mode")
            
        except Exception as e:
            error_msg = f"Error calculating charge movement: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.charge_status.set(error_msg)
            app_logger.error(error_msg)
    
    def plot_charge_movement(self, charge_movement):
        """Plot charge movement results"""
        self.charge_ax.clear()
        
        if charge_movement is not None:
            time_axis = np.arange(len(charge_movement)) * self.param_vars["sampling_interval"].get()
            self.charge_ax.plot(time_axis, charge_movement, 'r-', linewidth=2, label='Charge Movement')
            
            # Also plot original current for comparison
            if self.current_data is not None:
                orig_time = np.arange(len(self.current_data)) * self.param_vars["sampling_interval"].get()
                self.charge_ax.plot(orig_time, self.current_data, 'b-', alpha=0.5, label='Original Current')
            
            self.charge_ax.set_xlabel('Time (ms)')
            self.charge_ax.set_ylabel('Current (nA)')
            self.charge_ax.set_title(f'Charge Movement - {self.correction_mode.get().title()} Mode')
            self.charge_ax.grid(True, alpha=0.3)
            self.charge_ax.legend()
        
        self.charge_canvas.draw()
    
    def plot_in_main(self):
        """Plot charge movement in main application window"""
        try:
            if self.processor.charge_movement is None:
                messagebox.showwarning("Warning", "No charge movement calculated")
                return
            
            # Update main app data
            self.app.filtered_data = self.processor.charge_movement.copy()
            time_axis = np.arange(len(self.processor.charge_movement)) * self.param_vars["sampling_interval"].get()
            self.app.time_data = time_axis
            
            # Update main plot
            self.app.update_plot()
            
            messagebox.showinfo("Success", "Charge movement plotted in main window")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting in main window: {str(e)}")
    
    def update_summary(self):
        """Update the analysis summary"""
        summary = self.processor.get_processing_summary()
        
        summary_text = "=" * 60 + "\n"
        summary_text += "SIGNAL ANALYZER - ChaMa Analysis Summary\n"
        summary_text += "=" * 60 + "\n\n"
        
        # Protocol parameters
        summary_text += "PROTOCOL PARAMETERS:\n"
        summary_text += "-" * 30 + "\n"
        for param, value in summary.get("protocol_parameters", {}).items():
            summary_text += f"{param.replace('_', ' ').title()}: {value}\n"
        summary_text += "\n"
        
        # Processing status
        summary_text += "PROCESSING STATUS:\n"
        summary_text += "-" * 30 + "\n"
        summary_text += f"Negative Control: {'✓' if summary['has_negative_control'] else '✗'}\n"
        summary_text += f"Stored Control: {'✓' if summary['has_stored_control'] else '✗'}\n"
        summary_text += f"Charge Movement: {'✓' if summary['has_charge_movement'] else '✗'}\n"
        summary_text += f"Ohmic Fit: {'✓' if summary['has_ohmic_params'] else '✗'}\n"
        summary_text += "\n"
        
        # Statistics
        if summary['has_negative_control']:
            stats = summary.get('negative_control_stats', {})
            summary_text += "NEGATIVE CONTROL STATISTICS:\n"
            summary_text += "-" * 30 + "\n"
            summary_text += f"Length: {summary.get('negative_control_length', 'N/A')} points\n"
            summary_text += f"Mean: {stats.get('mean', 0):.6f} nA\n"
            summary_text += f"Std Dev: {stats.get('std', 0):.6f} nA\n"
            summary_text += f"Range: {stats.get('min', 0):.6f} to {stats.get('max', 0):.6f} nA\n"
            summary_text += "\n"
        
        if summary['has_ohmic_params']:
            ohmic = summary.get('ohmic_fit', {})
            summary_text += "OHMIC FIT PARAMETERS:\n"
            summary_text += "-" * 30 + "\n"
            summary_text += f"Slope: {ohmic.get('slope', 0):.8f} nA/point\n"
            summary_text += f"Intercept: {ohmic.get('intercept', 0):.6f} nA\n"
            summary_text += "\n"
        
        # Current correction mode
        summary_text += "CURRENT SETTINGS:\n"
        summary_text += "-" * 30 + "\n"
        summary_text += f"Correction Mode: {self.correction_mode.get().title()}\n"
        summary_text += f"Auto Baseline: {'Yes' if self.auto_baseline.get() else 'No'}\n"
        summary_text += f"Show Fit: {'Yes' if self.show_fit.get() else 'No'}\n"
        
        # Clear and insert new summary
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', summary_text)
    
    def load_protocol(self):
        """Load protocol parameters from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Protocol Parameters",
                filetypes=[("Protocol files", "*.prot"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'rb') as f:
                    params = pickle.load(f)
                
                for param_name, value in params.items():
                    if param_name in self.param_vars:
                        self.param_vars[param_name].set(value)
                
                messagebox.showinfo("Success", "Protocol parameters loaded successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading protocol: {str(e)}")
    
    def save_protocol(self):
        """Save current protocol parameters to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Protocol Parameters",
                defaultextension=".prot",
                filetypes=[("Protocol files", "*.prot"), ("All files", "*.*")]
            )
            
            if filename:
                params = {name: var.get() for name, var in self.param_vars.items()}
                
                with open(filename, 'wb') as f:
                    pickle.dump(params, f)
                
                messagebox.showinfo("Success", "Protocol parameters saved successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving protocol: {str(e)}")
    
    def reset_protocol(self):
        """Reset protocol parameters to defaults"""
        # Reset to default values (same as initialization)
        defaults = {
            "baseline_duration": 10.0,
            "neg_control_on_duration": 25.0,
            "neg_control_off_duration": 25.0,
            "test_on_duration": 20.0,
            "test_off_duration": 20.0,
            "sampling_interval": 0.1,
            "neg_control_v1": -80.0,
            "neg_control_v2": 0.0,
            "test_pulse_v1": -80.0,
            "test_pulse_v2": 0.0,
            "test_pulse_v3": -80.0,
            "num_control_traces": 3
        }
        
        for param, default in defaults.items():
            if param in self.param_vars:
                self.param_vars[param].set(default)
        
        messagebox.showinfo("Success", "Protocol parameters reset to defaults")
    
    def load_stored_control(self):
        """Load stored negative control from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Stored Negative Control",
                filetypes=[("Negative control files", "*.lnc"), ("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if filename:
                if filename.endswith('.npy'):
                    stored_control = np.load(filename)
                else:
                    # Load from text file (ChaMa .lnc format)
                    stored_control = np.loadtxt(filename)
                
                if self.processor.negative_control is not None:
                    # Add to existing control
                    self.processor.add_stored_negative_control(stored_control)
                    self.plot_negative_control(self.processor.negative_control)
                    messagebox.showinfo("Success", "Stored negative control added successfully")
                else:
                    # Use as current control
                    self.processor.negative_control = stored_control
                    self.plot_negative_control(stored_control)
                    messagebox.showinfo("Success", "Stored negative control loaded successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading stored control: {str(e)}")
    
    def save_negative_control(self):
        """Save current negative control to file"""
        try:
            if self.processor.negative_control is None:
                messagebox.showwarning("Warning", "No negative control to save")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Save Negative Control",
                defaultextension=".lnc",
                filetypes=[("Negative control files", "*.lnc"), ("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if filename:
                if filename.endswith('.npy'):
                    np.save(filename, self.processor.negative_control)
                else:
                    np.savetxt(filename, self.processor.negative_control)
                
                messagebox.showinfo("Success", "Negative control saved successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving negative control: {str(e)}")
    
    def export_charge_movement(self):
        """Export charge movement data"""
        try:
            if self.processor.charge_movement is None:
                messagebox.showwarning("Warning", "No charge movement data to export")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Export Charge Movement",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if filename:
                time_axis = np.arange(len(self.processor.charge_movement)) * self.param_vars["sampling_interval"].get()
                
                if filename.endswith('.csv'):
                    data = np.column_stack((time_axis, self.processor.charge_movement))
                    np.savetxt(filename, data, delimiter=',', header='Time(ms),Current(nA)', comments='')
                else:
                    np.save(filename, self.processor.charge_movement)
                
                messagebox.showinfo("Success", "Charge movement data exported successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")
    
    def export_summary(self):
        """Export analysis summary to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Analysis Summary",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                summary_content = self.summary_text.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(summary_content)
                
                messagebox.showinfo("Success", "Summary exported successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting summary: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Generate Analysis Report",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                # Generate comprehensive report with plots
                import datetime
                
                report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Signal Analyzer ChaMa Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #2c3e50; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .parameter {{ margin: 5px 0; }}
        .status {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Signal Analyzer - ChaMa Analysis Report</h1>
    <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Analysis Summary</h2>
        <pre>{self.summary_text.get('1.0', tk.END)}</pre>
    </div>
    
    <h2>Processing Details</h2>
    <p>This report contains the results of ChaMa-style electrophysiology analysis including
    negative control processing, ohmic component removal, and charge movement calculation.</p>
    
</body>
</html>
                """
                
                with open(filename, 'w') as f:
                    f.write(report_html)
                
                messagebox.showinfo("Success", "Analysis report generated successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {str(e)}")


# Integration function to add ChaMa tab to main application
def add_chama_tab_to_app(app):
    """Add enhanced ChaMa tab to the main Signal Analyzer application"""
    try:
        chama_tab = ChamaTab(app.notebook, app)
        app.notebook.add(chama_tab.frame, text='ChaMa Analysis')
        app.tabs['chama'] = chama_tab
        app_logger.info("Enhanced ChaMa tab added to application")
        return chama_tab
    except Exception as e:
        app_logger.error(f"Failed to add ChaMa tab: {str(e)}")
        return None