import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
                            QDoubleSpinBox, QGroupBox, QTextEdit, QFileDialog,
                            QMessageBox, QProgressBar, QCheckBox, QTabWidget,
                            QSplitter, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QPalette
import logging

app_logger = logging.getLogger(__name__)

class NegativeControlPlot(FigureCanvas):
    """Custom matplotlib canvas for displaying negative control traces"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create subplots
        self.ax_main = self.fig.add_subplot(211)
        self.ax_capacitance = self.fig.add_subplot(212)
        
        self.fig.tight_layout(pad=3.0)
        
    def plot_negative_control(self, time_axis, control_trace, title="Negative Control"):
        """Plot the negative control trace"""
        self.ax_main.clear()
        self.ax_main.plot(time_axis, control_trace, 'b-', linewidth=2, label='Negative Control')
        self.ax_main.set_xlabel('Time (ms)')
        self.ax_main.set_ylabel('Current (nA/mV)')
        self.ax_main.set_title(title)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        
    def plot_capacitance_integration(self, time_axis, control_trace, on_duration):
        """Plot capacitance integration regions"""
        self.ax_capacitance.clear()
        
        # Plot the trace
        self.ax_capacitance.plot(time_axis, control_trace, 'b-', linewidth=2)
        
        # Mark ON and OFF regions
        on_mask = time_axis <= on_duration
        off_mask = time_axis > on_duration
        
        if np.any(on_mask):
            self.ax_capacitance.fill_between(time_axis[on_mask], 0, control_trace[on_mask], 
                                           alpha=0.3, color='green', label='ON Integration')
        if np.any(off_mask):
            self.ax_capacitance.fill_between(time_axis[off_mask], 0, control_trace[off_mask], 
                                           alpha=0.3, color='red', label='OFF Integration')
        
        self.ax_capacitance.set_xlabel('Time (ms)')
        self.ax_capacitance.set_ylabel('Current (nA/mV)')
        self.ax_capacitance.set_title('Capacitance Integration Regions')
        self.ax_capacitance.grid(True, alpha=0.3)
        self.ax_capacitance.legend()
        self.ax_capacitance.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
    def refresh(self):
        """Refresh the canvas"""
        self.draw()

class NegativeControlWidget(QWidget):
    """
    Simplified widget for negative control processing functionality.
    Designed for standalone demo use.
    """
    
    # Signals for communication
    control_processed = pyqtSignal(np.ndarray)
    capacitance_calculated = pyqtSignal(float, float, float)
    status_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor = None  # Will be set externally
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the simplified user interface"""
        layout = QVBoxLayout(self)
        
        # Processing controls
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Ohmic component removal
        ohmic_layout = QHBoxLayout()
        ohmic_layout.addWidget(QLabel("Ohmic Removal:"))
        self.ohmic_method_combo = QComboBox()
        self.ohmic_method_combo.addItems(["None", "Average", "Linear", "Exponential"])
        ohmic_layout.addWidget(self.ohmic_method_combo)
        controls_layout.addLayout(ohmic_layout)
        
        # Subtraction mode
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Subtraction Mode:"))
        self.subtraction_mode_combo = QComboBox()
        self.subtraction_mode_combo.addItems(["Trace", "ON", "OFF", "Average"])
        sub_layout.addWidget(self.subtraction_mode_combo)
        controls_layout.addLayout(sub_layout)
        
        layout.addWidget(controls_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QGridLayout(results_group)
        
        results_layout.addWidget(QLabel("C-On (nF):"), 0, 0)
        self.c_on_display = QLineEdit()
        self.c_on_display.setReadOnly(True)
        results_layout.addWidget(self.c_on_display, 0, 1)
        
        results_layout.addWidget(QLabel("C-Off (nF):"), 1, 0)
        self.c_off_display = QLineEdit()
        self.c_off_display.setReadOnly(True)
        results_layout.addWidget(self.c_off_display, 1, 1)
        
        results_layout.addWidget(QLabel("C-Average (nF):"), 2, 0)
        self.c_avg_display = QLineEdit()
        self.c_avg_display.setReadOnly(True)
        results_layout.addWidget(self.c_avg_display, 2, 1)
        
        layout.addWidget(results_group)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
    def set_processor(self, processor):
        """Set the processor instance"""
        self.processor = processor
    
    def update_capacitance_display(self, c_on, c_off, c_avg):
        """Update capacitance display"""
        self.c_on_display.setText(f"{c_on:.3f}")
        self.c_off_display.setText(f"{c_off:.3f}")
        self.c_avg_display.setText(f"{c_avg:.3f}")
    
    def get_subtraction_mode(self):
        """Get current subtraction mode"""
        return self.subtraction_mode_combo.currentText().lower()
    
    def get_ohmic_removal_method(self):
        """Get current ohmic removal method"""
        method = self.ohmic_method_combo.currentText().lower()
        return method if method != "none" else None
    
    def update_status(self, message):
        """Update status display"""
        self.status_label.setText(message)
        self.status_changed.emit(message)