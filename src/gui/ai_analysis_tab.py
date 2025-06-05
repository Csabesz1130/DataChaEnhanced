# src/gui/ai_analysis_tab.py

"""
AI Analysis Tab for the Signal Analyzer Application

This tab provides a dedicated interface for AI-powered integral analysis,
including automatic calculation, manual comparison, validation, and export functionality.

Features:
- One-click AI analysis with confidence scoring
- Interactive manual analysis with real-time range adjustment
- AI vs Manual validation with detailed error analysis
- Quality metrics and performance monitoring
- Excel-compatible export functionality
- Auto-detection of optimal integration ranges
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.logger import app_logger
from src.analysis.ai_integral_calculator import AIIntegralCalculator
from src.config.ai_config import AIConfig, AIErrorMessages


class AIAnalysisTab:
    """
    Dedicated AI Analysis Tab for intelligent integral calculation and validation.
    
    This tab provides comprehensive AI functionality separate from the main processing,
    allowing users to leverage machine learning for accurate and efficient analysis.
    """
    
    def __init__(self, parent, app_reference):
        """
        Initialize the AI Analysis Tab.
        
        Args:
            parent: Parent notebook widget
            app_reference: Reference to the main application
        """
        self.parent = parent
        self.app = app_reference
        
        # Create the main tab frame
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text="AI Analysis")
        
        # Initialize AI calculator
        self.ai_calculator = AIIntegralCalculator()
        
        # Initialize state variables
        self.ai_results = None
        self.manual_results = None
        self.validation_results = None
        self.current_processor = None
        
        # Integration range variables
        self.hyperpol_start = tk.IntVar(value=10)  # Excel 11-1 = 10
        self.hyperpol_end = tk.IntVar(value=210)   # Excel 210
        self.depol_start = tk.IntVar(value=210)    # Excel 211-1 = 210
        self.depol_end = tk.IntVar(value=410)      # Excel 410
        
        # Analysis control variables
        self.auto_update_manual = tk.BooleanVar(value=True)
        self.show_range_indicators = tk.BooleanVar(value=True)
        self.validation_tolerance = tk.DoubleVar(value=0.15)  # 15% tolerance
        
        # Results display variables
        self.ai_results_text = tk.StringVar(value="Run AI analysis to see results")
        self.manual_results_text = tk.StringVar(value="Adjust ranges to see manual results")
        self.validation_text = tk.StringVar(value="Run both analyses to see validation")
        self.status_text = tk.StringVar(value="Ready for AI analysis")
        
        # Setup the user interface
        self.setup_ui()
        
        # Bind range variables to auto-update
        self.setup_auto_update_bindings()
        
        app_logger.info("AI Analysis tab initialized successfully")
    
    def setup_ui(self):
        """Setup the complete user interface for AI analysis."""
        # Create main sections
        self.create_header_section()
        self.create_ai_analysis_section()
        self.create_manual_analysis_section()
        self.create_validation_section()
        self.create_export_section()
        self.create_status_section()
        
    def create_header_section(self):
        """Create the header with title and quick actions."""
        header_frame = ttk.LabelFrame(self.frame, text="AI-Powered Integral Analysis", padding="10 5 10 5")
        header_frame.pack(fill='x', padx=5, pady=5)
        
        # Title and description
        title_label = ttk.Label(
            header_frame, 
            text="Automatic integral calculation based on Excel analysis patterns",
            font=('TkDefaultFont', 9, 'italic')
        )
        title_label.pack(pady=2)
        
        # Quick action buttons
        quick_actions_frame = ttk.Frame(header_frame)
        quick_actions_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            quick_actions_frame, 
            text="🧠 Run AI Analysis", 
            command=self.run_ai_analysis,
            style="Accent.TButton"
        ).pack(side='left', padx=5)
        
        ttk.Button(
            quick_actions_frame, 
            text="🔄 Run Manual Analysis", 
            command=self.run_manual_analysis
        ).pack(side='left', padx=5)
        
        ttk.Button(
            quick_actions_frame, 
            text="✓ Validate AI vs Manual", 
            command=self.run_validation
        ).pack(side='left', padx=5)
        
        ttk.Button(
            quick_actions_frame, 
            text="⚙️ Auto-Optimize Ranges", 
            command=self.auto_optimize_ranges
        ).pack(side='left', padx=5)
    
    def create_ai_analysis_section(self):
        """Create the AI analysis section with automatic calculation."""
        ai_frame = ttk.LabelFrame(self.frame, text="AI Analysis - Automatic Calculation", padding="5 5 5 5")
        ai_frame.pack(fill='x', padx=5, pady=5)
        
        # AI control frame
        ai_control_frame = ttk.Frame(ai_frame)
        ai_control_frame.pack(fill='x', pady=5)
        
        # Main AI analysis button
        ai_button = ttk.Button(
            ai_control_frame,
            text="🧠 Run AI Analysis",
            command=self.run_ai_analysis,
            style="Accent.TButton",
            width=20
        )
        ai_button.pack(side='left', padx=5)
        
        # AI configuration options
        config_frame = ttk.Frame(ai_control_frame)
        config_frame.pack(side='left', fill='x', expand=True, padx=10)
        
        ttk.Checkbutton(
            config_frame,
            text="Auto-optimize ranges",
            variable=tk.BooleanVar(value=True)
        ).pack(side='left', padx=5)
        
        ttk.Checkbutton(
            config_frame,
            text="Excel-compatible mode",
            variable=tk.BooleanVar(value=True)
        ).pack(side='left', padx=5)
        
        # AI results display
        ai_results_frame = ttk.LabelFrame(ai_frame, text="AI Results")
        ai_results_frame.pack(fill='x', pady=5)
        
        # Results text with scrolling capability
        ai_results_text_widget = tk.Text(
            ai_results_frame, 
            height=6, 
            wrap=tk.WORD,
            font=('Consolas', 9),
            relief='sunken',
            borderwidth=1
        )
        ai_results_text_widget.pack(fill='x', padx=5, pady=5)
        self.ai_results_widget = ai_results_text_widget
        
        # Quality metrics frame
        quality_frame = ttk.Frame(ai_frame)
        quality_frame.pack(fill='x', pady=5)
        
        ttk.Label(quality_frame, text="Quality Metrics:", font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        
        self.confidence_label = ttk.Label(quality_frame, text="Confidence: --")
        self.confidence_label.pack(side='left', padx=10)
        
        self.quality_label = ttk.Label(quality_frame, text="Signal Quality: --")
        self.quality_label.pack(side='left', padx=10)
        
        self.processing_time_label = ttk.Label(quality_frame, text="Processing Time: --")
        self.processing_time_label.pack(side='left', padx=10)
    
    def create_manual_analysis_section(self):
        """Create the manual analysis section with interactive ranges."""
        manual_frame = ttk.LabelFrame(self.frame, text="Manual Analysis - Interactive Range Selection", padding="5 5 5 5")
        manual_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Manual control frame
        manual_control_frame = ttk.Frame(manual_frame)
        manual_control_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            manual_control_frame,
            text="🔄 Update Manual Analysis",
            command=self.run_manual_analysis,
            width=20
        ).pack(side='left', padx=5)
        
        ttk.Checkbutton(
            manual_control_frame,
            text="Auto-update on range change",
            variable=self.auto_update_manual
        ).pack(side='left', padx=10)
        
        ttk.Checkbutton(
            manual_control_frame,
            text="Show range indicators on plot",
            variable=self.show_range_indicators,
            command=self.update_plot_indicators
        ).pack(side='left', padx=10)
        
        # Range selection frame
        ranges_frame = ttk.LabelFrame(manual_frame, text="Integration Ranges (Excel-compatible)")
        ranges_frame.pack(fill='x', pady=5)
        
        # Create two columns for hyperpol and depol
        hyperpol_frame = ttk.LabelFrame(ranges_frame, text="Hyperpolarization (Blue)", padding="5 5 5 5")
        hyperpol_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        depol_frame = ttk.LabelFrame(ranges_frame, text="Depolarization (Red)", padding="5 5 5 5")
        depol_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Hyperpolarization controls
        self.create_range_controls(hyperpol_frame, "hyperpol", "Hyperpol", 0, 250)
        
        # Depolarization controls  
        self.create_range_controls(depol_frame, "depol", "Depol", 150, 450)
        
        # Manual results display
        manual_results_frame = ttk.LabelFrame(manual_frame, text="Manual Results")
        manual_results_frame.pack(fill='x', pady=5)
        
        manual_results_text_widget = tk.Text(
            manual_results_frame,
            height=4,
            wrap=tk.WORD,
            font=('Consolas', 9),
            relief='sunken',
            borderwidth=1
        )
        manual_results_text_widget.pack(fill='x', padx=5, pady=5)
        self.manual_results_widget = manual_results_text_widget
        
        # Real-time integral display
        integral_display_frame = ttk.Frame(manual_frame)
        integral_display_frame.pack(fill='x', pady=5)
        
        self.hyperpol_integral_label = ttk.Label(
            integral_display_frame, 
            text="Hyperpol Integral: --", 
            font=('TkDefaultFont', 10, 'bold'),
            foreground='blue'
        )
        self.hyperpol_integral_label.pack(side='left', padx=10)
        
        self.depol_integral_label = ttk.Label(
            integral_display_frame, 
            text="Depol Integral: --", 
            font=('TkDefaultFont', 10, 'bold'),
            foreground='red'
        )
        self.depol_integral_label.pack(side='left', padx=10)
    
    def create_range_controls(self, parent_frame, range_type, label_prefix, min_val, max_val):
        """Create range control widgets for a specific range type."""
        # Start range control
        start_frame = ttk.Frame(parent_frame)
        start_frame.pack(fill='x', pady=2)
        
        ttk.Label(start_frame, text=f"{label_prefix} Start:").pack(side='left')
        
        start_var = getattr(self, f"{range_type}_start")
        start_scale = ttk.Scale(
            start_frame,
            from_=min_val,
            to=max_val-1,
            variable=start_var,
            orient='horizontal',
            command=lambda v: self.on_range_change(range_type, 'start', v)
        )
        start_scale.pack(side='left', fill='x', expand=True, padx=5)
        
        start_value_label = ttk.Label(start_frame, text=str(start_var.get()), width=5)
        start_value_label.pack(side='right')
        setattr(self, f"{range_type}_start_label", start_value_label)
        
        # End range control
        end_frame = ttk.Frame(parent_frame)
        end_frame.pack(fill='x', pady=2)
        
        ttk.Label(end_frame, text=f"{label_prefix} End:").pack(side='left')
        
        end_var = getattr(self, f"{range_type}_end")
        end_scale = ttk.Scale(
            end_frame,
            from_=min_val+1,
            to=max_val,
            variable=end_var,
            orient='horizontal',
            command=lambda v: self.on_range_change(range_type, 'end', v)
        )
        end_scale.pack(side='left', fill='x', expand=True, padx=5)
        
        end_value_label = ttk.Label(end_frame, text=str(end_var.get()), width=5)
        end_value_label.pack(side='right')
        setattr(self, f"{range_type}_end_label", end_value_label)
        
        # Direct entry fields
        entry_frame = ttk.Frame(parent_frame)
        entry_frame.pack(fill='x', pady=5)
        
        ttk.Label(entry_frame, text="Direct entry:").pack(side='left')
        
        start_entry = ttk.Entry(entry_frame, width=6)
        start_entry.pack(side='left', padx=2)
        start_entry.insert(0, str(start_var.get()))
        start_entry.bind('<Return>', lambda e: self.set_range_from_entry(range_type, 'start', start_entry))
        
        ttk.Label(entry_frame, text="to").pack(side='left', padx=2)
        
        end_entry = ttk.Entry(entry_frame, width=6)
        end_entry.pack(side='left', padx=2)
        end_entry.insert(0, str(end_var.get()))
        end_entry.bind('<Return>', lambda e: self.set_range_from_entry(range_type, 'end', end_entry))
        
        ttk.Button(
            entry_frame,
            text="Set",
            command=lambda: [
                self.set_range_from_entry(range_type, 'start', start_entry),
                self.set_range_from_entry(range_type, 'end', end_entry)
            ]
        ).pack(side='left', padx=5)
        
        # Store entry widgets for updates
        setattr(self, f"{range_type}_start_entry", start_entry)
        setattr(self, f"{range_type}_end_entry", end_entry)
        
        # Time display
        time_label = ttk.Label(
            parent_frame, 
            text=f"Time: {start_var.get() * 0.5:.1f} - {end_var.get() * 0.5:.1f} ms",
            font=('TkDefaultFont', 8, 'italic')
        )
        time_label.pack(pady=2)
        setattr(self, f"{range_type}_time_label", time_label)
    
    def create_validation_section(self):
        """Create the validation section for AI vs Manual comparison."""
        validation_frame = ttk.LabelFrame(self.frame, text="Validation - AI vs Manual Comparison", padding="5 5 5 5")
        validation_frame.pack(fill='x', padx=5, pady=5)
        
        # Validation controls
        validation_control_frame = ttk.Frame(validation_frame)
        validation_control_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            validation_control_frame,
            text="✓ Validate AI vs Manual",
            command=self.run_validation,
            width=20
        ).pack(side='left', padx=5)
        
        # Tolerance setting
        tolerance_frame = ttk.Frame(validation_control_frame)
        tolerance_frame.pack(side='left', padx=10)
        
        ttk.Label(tolerance_frame, text="Tolerance:").pack(side='left')
        tolerance_scale = ttk.Scale(
            tolerance_frame,
            from_=0.05,
            to=0.50,
            variable=self.validation_tolerance,
            orient='horizontal',
            length=100,
            command=self.update_tolerance_display
        )
        tolerance_scale.pack(side='left', padx=5)
        
        self.tolerance_label = ttk.Label(tolerance_frame, text="15%", width=5)
        self.tolerance_label.pack(side='left')
        
        # Validation results display
        validation_results_frame = ttk.LabelFrame(validation_frame, text="Validation Results")
        validation_results_frame.pack(fill='x', pady=5)
        
        # Error display
        error_display_frame = ttk.Frame(validation_results_frame)
        error_display_frame.pack(fill='x', pady=5)
        
        self.hyperpol_error_label = ttk.Label(error_display_frame, text="Hyperpol Error: --")
        self.hyperpol_error_label.pack(side='left', padx=10)
        
        self.depol_error_label = ttk.Label(error_display_frame, text="Depol Error: --")
        self.depol_error_label.pack(side='left', padx=10)
        
        self.overall_status_label = ttk.Label(
            error_display_frame, 
            text="Status: --", 
            font=('TkDefaultFont', 10, 'bold')
        )
        self.overall_status_label.pack(side='right', padx=10)
        
        # Detailed validation text
        validation_text_widget = tk.Text(
            validation_results_frame,
            height=3,
            wrap=tk.WORD,
            font=('Consolas', 9),
            relief='sunken',
            borderwidth=1
        )
        validation_text_widget.pack(fill='x', padx=5, pady=5)
        self.validation_widget = validation_text_widget
    
    def create_export_section(self):
        """Create the export section for saving results."""
        export_frame = ttk.LabelFrame(self.frame, text="Export & Save Results", padding="5 5 5 5")
        export_frame.pack(fill='x', padx=5, pady=5)
        
        export_buttons_frame = ttk.Frame(export_frame)
        export_buttons_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            export_buttons_frame,
            text="📊 Export Excel Format",
            command=self.export_excel_format,
            width=18
        ).pack(side='left', padx=5)
        
        ttk.Button(
            export_buttons_frame,
            text="📈 Export Plot",
            command=self.export_plot,
            width=15
        ).pack(side='left', padx=5)
        
        ttk.Button(
            export_buttons_frame,
            text="💾 Save AI Config",
            command=self.save_ai_config,
            width=15
        ).pack(side='left', padx=5)
        
        ttk.Button(
            export_buttons_frame,
            text="📋 Copy Results",
            command=self.copy_results_to_clipboard,
            width=15
        ).pack(side='left', padx=5)
    
    def create_status_section(self):
        """Create the status section for monitoring and feedback."""
        status_frame = ttk.Frame(self.frame)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Status label
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_text,
            font=('TkDefaultFont', 9, 'italic')
        )
        status_label.pack(side='left')
        
        # Progress indicator (could be expanded to actual progress bar)
        self.progress_label = ttk.Label(status_frame, text="")
        self.progress_label.pack(side='right')
    
    def setup_auto_update_bindings(self):
        """Setup automatic update bindings for range variables."""
        # Bind range variables to auto-update functions
        self.hyperpol_start.trace_add("write", lambda *args: self.update_range_displays('hyperpol'))
        self.hyperpol_end.trace_add("write", lambda *args: self.update_range_displays('hyperpol'))
        self.depol_start.trace_add("write", lambda *args: self.update_range_displays('depol'))
        self.depol_end.trace_add("write", lambda *args: self.update_range_displays('depol'))
        
        # Bind tolerance variable
        self.validation_tolerance.trace_add("write", lambda *args: self.update_tolerance_display())
    
    def run_ai_analysis(self):
        """Run the AI analysis with automatic integral calculation."""
        try:
            app_logger.info("Starting AI analysis")
            self.update_status("Running AI analysis...")
            
            # Get the current processor
            processor = self.get_current_processor()
            if not processor:
                self.show_error("Please process action potentials first")
                return
            
            # Run AI analysis
            start_time = time.time()
            
            self.ai_results = self.ai_calculator.analyze_action_potential(
                processor=processor,
                enable_auto_optimization=True
            )
            
            processing_time = time.time() - start_time
            
            # Update UI with results
            self.update_ai_results_display()
            self.update_quality_metrics_display()
            
            # Update plot if it exists
            self.update_plot_indicators()
            
            self.update_status(f"AI analysis completed in {processing_time:.2f}s")
            app_logger.info(f"AI analysis completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"AI analysis failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
            self.update_status("AI analysis failed")
    
    def run_manual_analysis(self):
        """Run manual analysis with current range settings."""
        try:
            app_logger.info("Starting manual analysis")
            self.update_status("Running manual analysis...")
            
            # Get the current processor
            processor = self.get_current_processor()
            if not processor:
                self.show_error("Please process action potentials first")
                return
            
            # Get current range settings
            hyperpol_range = (self.hyperpol_start.get(), self.hyperpol_end.get())
            depol_range = (self.depol_start.get(), self.depol_end.get())
            
            # Validate ranges
            if not self.validate_ranges(hyperpol_range, depol_range, processor):
                return
            
            # Run manual calculation
            self.manual_results = self.ai_calculator.calculate_manual_integrals(
                processor=processor,
                hyperpol_range=hyperpol_range,
                depol_range=depol_range
            )
            
            # Update UI with results
            self.update_manual_results_display()
            self.update_integral_labels()
            
            # Update plot indicators
            self.update_plot_indicators()
            
            self.update_status("Manual analysis completed")
            app_logger.info("Manual analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Manual analysis failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
            self.update_status("Manual analysis failed")
    
    def run_validation(self):
        """Run validation comparing AI vs Manual results."""
        try:
            if not self.ai_results:
                self.show_error("Please run AI analysis first")
                return
            
            if not self.manual_results:
                self.show_error("Please run manual analysis first")
                return
            
            app_logger.info("Starting AI vs Manual validation")
            self.update_status("Running validation...")
            
            # Run validation
            tolerance = self.validation_tolerance.get()
            self.validation_results = self.ai_calculator.validate_ai_vs_manual(
                ai_results=self.ai_results,
                manual_results=self.manual_results,
                tolerance=tolerance
            )
            
            # Update UI with validation results
            self.update_validation_display()
            
            self.update_status("Validation completed")
            app_logger.info("Validation completed successfully")
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
            self.update_status("Validation failed")
    
    def auto_optimize_ranges(self):
        """Automatically optimize integration ranges using AI."""
        try:
            app_logger.info("Starting automatic range optimization")
            self.update_status("Optimizing ranges...")
            
            # Get the current processor
            processor = self.get_current_processor()
            if not processor:
                self.show_error("Please process action potentials first")
                return
            
            # Run optimization
            optimization_result = self.ai_calculator.optimize_integration_ranges(
                processor=processor,
                method='adaptive'
            )
            
            # Update range variables with optimized values
            if 'hyperpol_range' in optimization_result:
                hyperpol_range = optimization_result['hyperpol_range']
                self.hyperpol_start.set(hyperpol_range[0])
                self.hyperpol_end.set(hyperpol_range[1])
            
            if 'depol_range' in optimization_result:
                depol_range = optimization_result['depol_range']
                self.depol_start.set(depol_range[0])
                self.depol_end.set(depol_range[1])
            
            # Update displays
            self.update_range_displays('hyperpol')
            self.update_range_displays('depol')
            
            # Auto-run manual analysis with optimized ranges
            if self.auto_update_manual.get():
                self.run_manual_analysis()
            
            confidence = optimization_result.get('confidence', 0)
            self.update_status(f"Ranges optimized (confidence: {confidence:.1%})")
            
            messagebox.showinfo(
                "Range Optimization", 
                f"Ranges optimized with {confidence:.1%} confidence\n"
                f"Hyperpol: {hyperpol_range[0]}-{hyperpol_range[1]}\n"
                f"Depol: {depol_range[0]}-{depol_range[1]}"
            )
            
        except Exception as e:
            error_msg = f"Range optimization failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
            self.update_status("Range optimization failed")
    
    def get_current_processor(self):
        """Get the current action potential processor from the main app."""
        if hasattr(self.app, 'action_potential_processor'):
            return self.app.action_potential_processor
        
        app_logger.warning("No action potential processor found")
        return None
    
    def validate_ranges(self, hyperpol_range: Tuple[int, int], depol_range: Tuple[int, int], 
                       processor) -> bool:
        """Validate that the specified ranges are valid for the current data."""
        try:
            # Check range validity
            if hyperpol_range[0] >= hyperpol_range[1]:
                self.show_error("Invalid hyperpolarization range: start must be less than end")
                return False
            
            if depol_range[0] >= depol_range[1]:
                self.show_error("Invalid depolarization range: start must be less than end")
                return False
            
            # Check data bounds
            hyperpol_length = len(processor.modified_hyperpol) if hasattr(processor, 'modified_hyperpol') else 0
            depol_length = len(processor.modified_depol) if hasattr(processor, 'modified_depol') else 0
            
            if hyperpol_range[1] > hyperpol_length:
                self.show_error(f"Hyperpolarization range exceeds data length ({hyperpol_length})")
                return False
            
            if depol_range[1] > depol_length:
                self.show_error(f"Depolarization range exceeds data length ({depol_length})")
                return False
            
            return True
            
        except Exception as e:
            self.show_error(f"Range validation failed: {str(e)}")
            return False
    
    def on_range_change(self, range_type: str, range_part: str, value):
        """Handle range slider changes."""
        try:
            # Update the corresponding label
            int_value = int(float(value))
            label = getattr(self, f"{range_type}_{range_part}_label")
            label.config(text=str(int_value))
            
            # Update entry fields
            entry = getattr(self, f"{range_type}_{range_part}_entry")
            entry.delete(0, tk.END)
            entry.insert(0, str(int_value))
            
            # Update time display
            self.update_range_displays(range_type)
            
            # Auto-update manual analysis if enabled
            if self.auto_update_manual.get():
                self.app.after_idle(self.run_manual_analysis)
                
        except Exception as e:
            app_logger.error(f"Error handling range change: {str(e)}")
    
    def set_range_from_entry(self, range_type: str, range_part: str, entry_widget):
        """Set range value from direct entry."""
        try:
            value = int(entry_widget.get())
            range_var = getattr(self, f"{range_type}_{range_part}")
            range_var.set(value)
            
            # Update displays will be triggered by the variable trace
            
        except ValueError:
            self.show_error("Please enter a valid integer")
            # Reset entry to current value
            current_value = getattr(self, f"{range_type}_{range_part}").get()
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, str(current_value))
    
    def update_range_displays(self, range_type: str):
        """Update range displays including time labels."""
        try:
            start_var = getattr(self, f"{range_type}_start")
            end_var = getattr(self, f"{range_type}_end")
            time_label = getattr(self, f"{range_type}_time_label")
            
            start_val = start_var.get()
            end_val = end_var.get()
            
            # Update time display (0.5 ms per point)
            start_time = start_val * 0.5
            end_time = end_val * 0.5
            time_label.config(text=f"Time: {start_time:.1f} - {end_time:.1f} ms")
            
        except Exception as e:
            app_logger.error(f"Error updating range displays: {str(e)}")
    
    def update_tolerance_display(self, *args):
        """Update the tolerance percentage display."""
        tolerance_percent = self.validation_tolerance.get() * 100
        self.tolerance_label.config(text=f"{tolerance_percent:.0f}%")
    
    def update_ai_results_display(self):
        """Update the AI results display widget."""
        if not self.ai_results:
            return
        
        try:
            # Format AI results for display
            ai_integrals = self.ai_results['ai_integrals']
            confidence = self.ai_results['confidence_scores']
            processing_info = self.ai_results['processing_info']
            
            result_text = [
                "=== AI ANALYSIS RESULTS ===",
                f"Hyperpolarization Integral: {ai_integrals['hyperpol_integral']:.3f} pC",
                f"Depolarization Integral: {ai_integrals['depol_integral']:.3f} pC",
                "",
                f"Overall Confidence: {confidence['overall_confidence']:.1%} ({confidence['confidence_level']})",
                f"Processing Time: {processing_info['processing_time']:.2f} seconds",
                f"Excel Compatible: {processing_info['excel_compatible']}",
                "",
                "Integration Ranges Used:",
                f"  Hyperpol: {self.ai_results['integration_ranges']['hyperpol_range']}",
                f"  Depol: {self.ai_results['integration_ranges']['depol_range']}",
                f"  Source: {self.ai_results['integration_ranges']['source']}"
            ]
            
            # Update the text widget
            self.ai_results_widget.config(state='normal')
            self.ai_results_widget.delete(1.0, tk.END)
            self.ai_results_widget.insert(1.0, '\n'.join(result_text))
            self.ai_results_widget.config(state='disabled')
            
        except Exception as e:
            app_logger.error(f"Error updating AI results display: {str(e)}")
    
    def update_quality_metrics_display(self):
        """Update the quality metrics labels."""
        if not self.ai_results:
            return
        
        try:
            confidence = self.ai_results['confidence_scores']['overall_confidence']
            confidence_level = self.ai_results['confidence_scores']['confidence_level']
            
            quality_metrics = self.ai_results['quality_metrics']
            avg_quality = quality_metrics.get('combined', {}).get('average_quality', 0)
            
            processing_time = self.ai_results['processing_info']['processing_time']
            
            # Update labels
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%} ({confidence_level})",
                foreground=self.get_confidence_color(confidence)
            )
            
            self.quality_label.config(
                text=f"Signal Quality: {avg_quality:.1%}",
                foreground=self.get_quality_color(avg_quality)
            )
            
            self.processing_time_label.config(
                text=f"Processing Time: {processing_time:.2f}s",
                foreground='green' if processing_time < 2.0 else 'orange'
            )
            
        except Exception as e:
            app_logger.error(f"Error updating quality metrics: {str(e)}")
    
    def update_manual_results_display(self):
        """Update the manual results display widget."""
        if not self.manual_results:
            return
        
        try:
            result_text = [
                "=== MANUAL ANALYSIS RESULTS ===",
                f"Hyperpolarization Integral: {self.manual_results['hyperpol_integral']:.3f} pC",
                f"Depolarization Integral: {self.manual_results['depol_integral']:.3f} pC",
                "",
                "Integration Ranges:",
                f"  Hyperpol: {self.manual_results['hyperpol_range']} "
                f"({self.manual_results['range_lengths']['hyperpol']} points)",
                f"  Depol: {self.manual_results['depol_range']} "
                f"({self.manual_results['range_lengths']['depol']} points)",
                "",
                "Time Spans:",
                f"  Hyperpol: {self.manual_results['time_spans']['hyperpol'][0]:.1f} - "
                f"{self.manual_results['time_spans']['hyperpol'][1]:.1f} ms",
                f"  Depol: {self.manual_results['time_spans']['depol'][0]:.1f} - "
                f"{self.manual_results['time_spans']['depol'][1]:.1f} ms"
            ]
            
            # Update the text widget
            self.manual_results_widget.config(state='normal')
            self.manual_results_widget.delete(1.0, tk.END)
            self.manual_results_widget.insert(1.0, '\n'.join(result_text))
            self.manual_results_widget.config(state='disabled')
            
        except Exception as e:
            app_logger.error(f"Error updating manual results display: {str(e)}")
    
    def update_integral_labels(self):
        """Update the real-time integral display labels."""
        if not self.manual_results:
            return
        
        try:
            hyperpol_integral = self.manual_results['hyperpol_integral']
            depol_integral = self.manual_results['depol_integral']
            
            self.hyperpol_integral_label.config(
                text=f"Hyperpol Integral: {hyperpol_integral:.3f} pC"
            )
            
            self.depol_integral_label.config(
                text=f"Depol Integral: {depol_integral:.3f} pC"
            )
            
        except Exception as e:
            app_logger.error(f"Error updating integral labels: {str(e)}")
    
    def update_validation_display(self):
        """Update the validation results display."""
        if not self.validation_results:
            return
        
        try:
            val_results = self.validation_results
            
            # Update error labels
            hyperpol_error = val_results['hyperpol_error_percent']
            depol_error = val_results['depol_error_percent']
            
            self.hyperpol_error_label.config(
                text=f"Hyperpol Error: {hyperpol_error:.1f}%",
                foreground=self.get_error_color(hyperpol_error, val_results['tolerance_used'])
            )
            
            self.depol_error_label.config(
                text=f"Depol Error: {depol_error:.1f}%",
                foreground=self.get_error_color(depol_error, val_results['tolerance_used'])
            )
            
            # Update overall status
            status = val_results['validation_summary']['status']
            status_color = 'green' if status == 'PASS' else 'red'
            
            self.overall_status_label.config(
                text=f"Status: {status}",
                foreground=status_color
            )
            
            # Update detailed validation text
            validation_text = [
                "=== VALIDATION RESULTS ===",
                f"Hyperpolarization Error: {hyperpol_error:.2f}% ({'PASS' if val_results['hyperpol_pass'] else 'FAIL'})",
                f"Depolarization Error: {depol_error:.2f}% ({'PASS' if val_results['depol_pass'] else 'FAIL'})",
                f"Tolerance Used: {val_results['tolerance_used']:.0f}%",
                f"Overall Status: {status}",
                f"Quality Score: {val_results['validation_summary']['quality_score']:.2f}",
                "",
                "Recommendations:"
            ]
            
            for recommendation in val_results.get('recommendations', []):
                validation_text.append(f"  • {recommendation}")
            
            self.validation_widget.config(state='normal')
            self.validation_widget.delete(1.0, tk.END)
            self.validation_widget.insert(1.0, '\n'.join(validation_text))
            self.validation_widget.config(state='disabled')
            
        except Exception as e:
            app_logger.error(f"Error updating validation display: {str(e)}")
    
    def update_plot_indicators(self):
        """Update range indicators on the main plot (if available)."""
        try:
            if not self.show_range_indicators.get():
                return
            
            # Try to get the main plot from the app
            if hasattr(self.app, 'ax') and self.app.ax is not None:
                ax = self.app.ax
                
                # Clear existing range indicators
                for child in ax.get_children():
                    if hasattr(child, 'get_label') and 'range_indicator' in str(child.get_label()):
                        child.remove()
                
                # Add new range indicators if we have results
                if self.manual_results:
                    self.add_range_indicators_to_plot(ax)
                
                # Refresh the plot
                if hasattr(self.app, 'canvas'):
                    self.app.canvas.draw_idle()
            
        except Exception as e:
            app_logger.debug(f"Could not update plot indicators: {str(e)}")
    
    def add_range_indicators_to_plot(self, ax):
        """Add range indicators to the specified matplotlib axis."""
        try:
            processor = self.get_current_processor()
            if not processor:
                return
            
            # Get time arrays
            if hasattr(processor, 'modified_hyperpol_times'):
                hyperpol_times = processor.modified_hyperpol_times * 1000  # Convert to ms
                hyperpol_start_time = hyperpol_times[self.hyperpol_start.get()]
                hyperpol_end_time = hyperpol_times[min(self.hyperpol_end.get()-1, len(hyperpol_times)-1)]
                
                ax.axvspan(hyperpol_start_time, hyperpol_end_time, 
                          alpha=0.3, color='blue', 
                          label='hyperpol_range_indicator')
            
            if hasattr(processor, 'modified_depol_times'):
                depol_times = processor.modified_depol_times * 1000  # Convert to ms
                depol_start_time = depol_times[self.depol_start.get()]
                depol_end_time = depol_times[min(self.depol_end.get()-1, len(depol_times)-1)]
                
                ax.axvspan(depol_start_time, depol_end_time, 
                          alpha=0.3, color='red', 
                          label='depol_range_indicator')
            
        except Exception as e:
            app_logger.debug(f"Error adding range indicators: {str(e)}")
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return 'green'
        elif confidence >= 0.5:
            return 'orange'
        else:
            return 'red'
    
    def get_quality_color(self, quality: float) -> str:
        """Get color based on quality level."""
        if quality >= 0.8:
            return 'green'
        elif quality >= 0.5:
            return 'orange'
        else:
            return 'red'
    
    def get_error_color(self, error_percent: float, tolerance_percent: float) -> str:
        """Get color based on error relative to tolerance."""
        if error_percent <= tolerance_percent:
            return 'green'
        elif error_percent <= tolerance_percent * 1.5:
            return 'orange'
        else:
            return 'red'
    
    def export_excel_format(self):
        """Export results in Excel-compatible format."""
        try:
            if not self.ai_results and not self.manual_results:
                self.show_error("No results to export. Please run analysis first.")
                return
            
            # Open save dialog
            filename = filedialog.asksaveasfilename(
                title="Export Excel-Compatible Results",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                initialname=f"ai_analysis_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if not filename:
                return
            
            # Prepare export content
            export_content = self.prepare_excel_export_content()
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(export_content)
            
            self.update_status(f"Results exported to {filename}")
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filename}")
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
    
    def prepare_excel_export_content(self) -> str:
        """Prepare comprehensive export content."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        content = [
            "=== AI-POWERED ACTION POTENTIAL ANALYSIS ===",
            f"Export Date: {timestamp}",
            f"Analysis Mode: Excel-Compatible",
            ""
        ]
        
        # AI Results
        if self.ai_results:
            ai_integrals = self.ai_results['ai_integrals']
            content.extend([
                "AI ANALYSIS:",
                f"  Hyperpolarization Integral: {ai_integrals['hyperpol_integral']:.3f} pC",
                f"  Depolarization Integral: {ai_integrals['depol_integral']:.3f} pC",
                f"  Confidence: {self.ai_results['confidence_scores']['overall_confidence']:.1%}",
                f"  Processing Time: {self.ai_results['processing_info']['processing_time']:.2f}s",
                ""
            ])
        
        # Manual Results
        if self.manual_results:
            content.extend([
                "MANUAL ANALYSIS:",
                f"  Hyperpolarization Integral: {self.manual_results['hyperpol_integral']:.3f} pC",
                f"  Depolarization Integral: {self.manual_results['depol_integral']:.3f} pC",
                f"  Hyperpol Range: {self.manual_results['hyperpol_range']}",
                f"  Depol Range: {self.manual_results['depol_range']}",
                ""
            ])
        
        # Validation Results
        if self.validation_results:
            val = self.validation_results
            content.extend([
                "VALIDATION:",
                f"  Hyperpol Error: {val['hyperpol_error_percent']:.2f}%",
                f"  Depol Error: {val['depol_error_percent']:.2f}%",
                f"  Overall Status: {val['validation_summary']['status']}",
                f"  Tolerance: {val['tolerance_used']:.0f}%",
                ""
            ])
        
        # Configuration
        content.extend([
            "CONFIGURATION:",
            f"  Excel Point Ranges: Hyperpol(11-210), Depol(211-410)",
            f"  Python Ranges: Hyperpol(10-210), Depol(210-410)",
            f"  Sampling Rate: 0.5 ms per point",
            f"  Integration Method: Trapezoidal with /2 correction",
            ""
        ])
        
        return "\n".join(content)
    
    def export_plot(self):
        """Export the current plot with range indicators."""
        try:
            if not hasattr(self.app, 'ax') or self.app.ax is None:
                self.show_error("No plot available to export")
                return
            
            # Open save dialog
            filename = filedialog.asksaveasfilename(
                title="Export Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                initialname=f"ai_analysis_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            if not filename:
                return
            
            # Save the plot
            fig = self.app.ax.get_figure()
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            self.update_status(f"Plot exported to {filename}")
            messagebox.showinfo("Export Complete", f"Plot exported to:\n{filename}")
            
        except Exception as e:
            error_msg = f"Plot export failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
    
    def save_ai_config(self):
        """Save current AI configuration settings."""
        try:
            # Open save dialog
            filename = filedialog.asksaveasfilename(
                title="Save AI Configuration",
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ],
                initialname=f"ai_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if not filename:
                return
            
            # Prepare configuration
            config = {
                "integration_ranges": {
                    "hyperpol_range": (self.hyperpol_start.get(), self.hyperpol_end.get()),
                    "depol_range": (self.depol_start.get(), self.depol_end.get())
                },
                "validation_settings": {
                    "tolerance": self.validation_tolerance.get(),
                    "auto_update": self.auto_update_manual.get()
                },
                "display_settings": {
                    "show_range_indicators": self.show_range_indicators.get()
                },
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save configuration
            import json
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.update_status(f"Configuration saved to {filename}")
            messagebox.showinfo("Save Complete", f"Configuration saved to:\n{filename}")
            
        except Exception as e:
            error_msg = f"Configuration save failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
    
    def copy_results_to_clipboard(self):
        """Copy results to system clipboard."""
        try:
            # Prepare clipboard content
            clipboard_content = self.prepare_clipboard_content()
            
            # Copy to clipboard
            self.app.clipboard_clear()
            self.app.clipboard_append(clipboard_content)
            
            self.update_status("Results copied to clipboard")
            messagebox.showinfo("Copy Complete", "Results copied to clipboard")
            
        except Exception as e:
            error_msg = f"Copy to clipboard failed: {str(e)}"
            app_logger.error(error_msg)
            self.show_error(error_msg)
    
    def prepare_clipboard_content(self) -> str:
        """Prepare content for clipboard."""
        content = []
        
        if self.ai_results:
            ai_integrals = self.ai_results['ai_integrals']
            content.append(f"AI: H={ai_integrals['hyperpol_integral']:.3f}pC, D={ai_integrals['depol_integral']:.3f}pC")
        
        if self.manual_results:
            content.append(f"Manual: H={self.manual_results['hyperpol_integral']:.3f}pC, D={self.manual_results['depol_integral']:.3f}pC")
        
        if self.validation_results:
            val = self.validation_results
            content.append(f"Validation: H_err={val['hyperpol_error_percent']:.1f}%, D_err={val['depol_error_percent']:.1f}%, Status={val['validation_summary']['status']}")
        
        return "\n".join(content)
    
    def update_status(self, message: str):
        """Update the status message."""
        self.status_text.set(message)
        app_logger.info(f"Status: {message}")
    
    def show_error(self, message: str):
        """Show error message to user."""
        messagebox.showerror("Error", message)
        app_logger.error(message)
    
    def on_tab_selected(self):
        """Called when this tab is selected."""
        # Refresh data if needed
        try:
            if hasattr(self.app, 'action_potential_processor'):
                processor = self.app.action_potential_processor
                if processor and hasattr(processor, 'modified_hyperpol'):
                    # Update range limits based on actual data length
                    self.update_range_limits(processor)
        except Exception as e:
            app_logger.debug(f"Error refreshing tab data: {str(e)}")
    
    def update_range_limits(self, processor):
        """Update range slider limits based on actual data."""
        try:
            if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
                hyperpol_length = len(processor.modified_hyperpol)
                # Update hyperpol sliders maximum values
                # Implementation would require access to the actual slider widgets
                
            if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
                depol_length = len(processor.modified_depol)
                # Update depol sliders maximum values
                # Implementation would require access to the actual slider widgets
                
        except Exception as e:
            app_logger.debug(f"Error updating range limits: {str(e)}")