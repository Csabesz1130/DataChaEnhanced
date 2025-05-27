"""Plot windows module for signal analysis"""

from src.gui.plot_windows.plot_window_base import PlotWindowBase
from src.gui.plot_windows.baseline_window import BaselineCorrectionWindow
from src.gui.plot_windows.normalization_window import NormalizationWindow
from src.gui.plot_windows.integration_window import IntegrationWindow

__all__ = [
    'PlotWindowBase',
    'BaselineCorrectionWindow',
    'NormalizationWindow',
    'IntegrationWindow'
]