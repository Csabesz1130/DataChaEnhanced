"""
Linear Fit Subtraction GUI Module for DataChaEnhanced
====================================================
Location: src/gui/linear_fit_subtraction_gui.py

This module provides GUI controls for the linear fit subtraction feature.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QPushButton, QLabel, QCheckBox, QTextEdit, 
                            QMessageBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)

class LinearFitSubtractionPanel(QWidget):
    """
    GUI panel for linear fit subtraction controls and display.
    """
    
    # Signals
    subtraction_requested = pyqtSignal(str)  # curve_type
    both_subtraction_requested = pyqtSignal()
    reset_requested = pyqtSignal(str)  # curve_type or 'all'
    plot_reload_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.subtractor = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Main controls group
        controls_group = QGroupBox("Linear Fit Subtraction")
        controls_layout = QVBoxLayout(controls_group)
        
        # Individual curve controls
        individual_layout = QHBoxLayout()
        
        # Hyperpol controls
        hyperpol_frame = QFrame()
        hyperpol_frame.setFrameStyle(QFrame.StyledPanel)
        hyperpol_layout = QVBoxLayout(hyperpol_frame)
        
        hyperpol_layout.addWidget(QLabel("Hyperpol Curve:"))
        self.hyperpol_subtract_btn = QPushButton("Subtract Linear Fit")
        self.hyperpol_subtract_btn.clicked.connect(lambda: self.subtraction_requested.emit('hyperpol'))
        hyperpol_layout.addWidget(self.hyperpol_subtract_btn)
        
        self.hyperpol_reset_btn = QPushButton("Reset")
        self.hyperpol_reset_btn.clicked.connect(lambda: self.reset_requested.emit('hyperpol'))
        hyperpol_layout.addWidget(self.hyperpol_reset_btn)
        
        individual_layout.addWidget(hyperpol_frame)
        
        # Depol controls
        depol_frame = QFrame()
        depol_frame.setFrameStyle(QFrame.StyledPanel)
        depol_layout = QVBoxLayout(depol_frame)
        
        depol_layout.addWidget(QLabel("Depol Curve:"))
        self.depol_subtract_btn = QPushButton("Subtract Linear Fit")
        self.depol_subtract_btn.clicked.connect(lambda: self.subtraction_requested.emit('depol'))
        depol_layout.addWidget(self.depol_subtract_btn)
        
        self.depol_reset_btn = QPushButton("Reset")
        self.depol_reset_btn.clicked.connect(lambda: self.reset_requested.emit('depol'))
        depol_layout.addWidget(self.depol_reset_btn)
        
        individual_layout.addWidget(depol_frame)
        
        controls_layout.addLayout(individual_layout)
        
        # Both curves control
        both_layout = QHBoxLayout()
        self.both_subtract_btn = QPushButton("Subtract Both Curves")
        self.both_subtract_btn.clicked.connect(self.both_subtraction_requested.emit)
        both_layout.addWidget(self.both_subtract_btn)
        
        self.both_reset_btn = QPushButton("Reset All")
        self.both_reset_btn.clicked.connect(lambda: self.reset_requested.emit('all'))
        both_layout.addWidget(self.both_reset_btn)
        
        controls_layout.addLayout(both_layout)
        
        # Reload plot button
        reload_layout = QHBoxLayout()
        self.reload_plot_btn = QPushButton("Reload Plot")
        self.reload_plot_btn.clicked.connect(self.plot_reload_requested.emit)
        reload_layout.addWidget(self.reload_plot_btn)
        
        controls_layout.addLayout(reload_layout)
        
        layout.addWidget(controls_group)
        
        # Results display group
        results_group = QGroupBox("Subtraction Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 9))
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: blue; }")
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
        # Initially disable buttons
        self.set_buttons_enabled(False)
    
    def set_subtractor(self, subtractor):
        """Set the linear fit subtractor instance."""
        self.subtractor = subtractor
        self.update_display()
    
    def set_buttons_enabled(self, enabled: bool):
        """Enable or disable all control buttons."""
        self.hyperpol_subtract_btn.setEnabled(enabled)
        self.hyperpol_reset_btn.setEnabled(enabled)
        self.depol_subtract_btn.setEnabled(enabled)
        self.depol_reset_btn.setEnabled(enabled)
        self.both_subtract_btn.setEnabled(enabled)
        self.both_reset_btn.setEnabled(enabled)
        self.reload_plot_btn.setEnabled(enabled)
    
    def update_display(self):
        """Update the results display with current information."""
        if not self.subtractor:
            self.results_text.clear()
            self.status_label.setText("No subtractor available")
            return
        
        # Get fit information
        hyperpol_info = self.subtractor.get_fit_info('hyperpol')
        depol_info = self.subtractor.get_fit_info('depol')
        
        # Get subtraction status
        hyperpol_subtracted = self.subtractor.get_subtracted_data('hyperpol') is not None
        depol_subtracted = self.subtractor.get_subtracted_data('depol') is not None
        
        # Build display text
        display_text = "Linear Fit Information:\n"
        display_text += "=" * 50 + "\n\n"
        
        # Hyperpol info
        display_text += "Hyperpol Curve:\n"
        if hyperpol_info:
            display_text += f"  Equation: {hyperpol_info['equation']}\n"
            display_text += f"  R² = {hyperpol_info['r_squared']:.4f}\n"
            display_text += f"  Time range: {hyperpol_info['start_time']:.3f} - {hyperpol_info['end_time']:.3f} s\n"
        else:
            display_text += "  No linear fit available\n"
        
        display_text += f"  Subtraction status: {'✓ Subtracted' if hyperpol_subtracted else '✗ Not subtracted'}\n\n"
        
        # Depol info
        display_text += "Depol Curve:\n"
        if depol_info:
            display_text += f"  Equation: {depol_info['equation']}\n"
            display_text += f"  R² = {depol_info['r_squared']:.4f}\n"
            display_text += f"  Time range: {depol_info['start_time']:.3f} - {depol_info['end_time']:.3f} s\n"
        else:
            display_text += "  No linear fit available\n"
        
        display_text += f"  Subtraction status: {'✓ Subtracted' if depol_subtracted else '✗ Not subtracted'}\n"
        
        self.results_text.setPlainText(display_text)
        
        # Update status
        if hyperpol_subtracted and depol_subtracted:
            self.status_label.setText("Both curves subtracted")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        elif hyperpol_subtracted or depol_subtracted:
            self.status_label.setText("Partial subtraction completed")
            self.status_label.setStyleSheet("QLabel { color: orange; }")
        else:
            self.status_label.setText("Ready for subtraction")
            self.status_label.setStyleSheet("QLabel { color: blue; }")
    
    def update_status(self, message: str, color: str = "blue"):
        """Update the status label with a message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"QLabel {{ color: {color}; }}")
    
    def show_error(self, message: str):
        """Show an error message."""
        QMessageBox.critical(self, "Error", message)
        self.update_status(f"Error: {message}", "red")
    
    def show_success(self, message: str):
        """Show a success message."""
        self.update_status(message, "green")
        # Update display after successful operation
        self.update_display()
    
    def on_subtraction_completed(self, curve_type: str, success: bool, message: str = ""):
        """Handle subtraction completion."""
        if success:
            self.show_success(f"Subtraction completed for {curve_type}")
        else:
            self.show_error(f"Subtraction failed for {curve_type}: {message}")
    
    def on_both_subtraction_completed(self, results: dict, success: bool, message: str = ""):
        """Handle both curves subtraction completion."""
        if success:
            completed_curves = list(results.keys())
            self.show_success(f"Subtraction completed for: {', '.join(completed_curves)}")
        else:
            self.show_error(f"Subtraction failed: {message}")
    
    def on_reset_completed(self, curve_type: str, success: bool, message: str = ""):
        """Handle reset completion."""
        if success:
            if curve_type == 'all':
                self.show_success("All subtraction data reset")
            else:
                self.show_success(f"Reset completed for {curve_type}")
        else:
            self.show_error(f"Reset failed for {curve_type}: {message}")
