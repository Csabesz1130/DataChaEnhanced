"""
Linear Fit Subtraction Integration Module for DataChaEnhanced
============================================================
Location: src/gui/linear_fit_subtraction_integration.py

This module integrates the linear fit subtraction feature with the main application.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple

from src.analysis.linear_fit_subtractor import LinearFitSubtractor
from src.gui.linear_fit_subtraction_gui import LinearFitSubtractionPanel

logger = logging.getLogger(__name__)

class LinearFitSubtractionIntegration:
    """
    Integrates linear fit subtraction with the main application.
    """
    
    def __init__(self, main_app):
        """
        Initialize the integration with the main application.
        
        Args:
            main_app: Reference to the main application instance
        """
        self.main_app = main_app
        self.subtractor = LinearFitSubtractor()
        self.panel = None
        self._setup_integration()
    
    def _setup_integration(self):
        """Set up the integration with the main application."""
        # Create the GUI panel
        self.panel = LinearFitSubtractionPanel()
        self.panel.set_subtractor(self.subtractor)
        
        # Connect signals
        self.panel.subtraction_requested.connect(self._on_subtraction_requested)
        self.panel.both_subtraction_requested.connect(self._on_both_subtraction_requested)
        self.panel.reset_requested.connect(self._on_reset_requested)
        self.panel.plot_reload_requested.connect(self._on_plot_reload_requested)
        
        logger.info("Linear fit subtraction integration initialized")
    
    def get_panel(self) -> LinearFitSubtractionPanel:
        """Get the GUI panel for this feature."""
        return self.panel
    
    def update_from_curve_fitting(self, curve_type: str, linear_params: Dict, 
                                 linear_curve: Dict, r_squared: float):
        """
        Update the subtractor with fitted curve data from the curve fitting manager.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            linear_params: Linear fit parameters
            linear_curve: Linear fit curve data
            r_squared: R-squared value
        """
        try:
            self.subtractor.set_fitted_curves(curve_type, linear_params, linear_curve, r_squared)
            
            # Update original data if available
            self._update_original_data(curve_type)
            
            # Update the panel display
            self.panel.update_display()
            
            # Enable buttons if we have data
            self._update_button_states()
            
            logger.info(f"Updated subtractor with {curve_type} linear fit data")
            
        except Exception as e:
            logger.error(f"Failed to update subtractor with {curve_type} data: {str(e)}")
            self.panel.show_error(f"Failed to update with {curve_type} data: {str(e)}")
    
    def _update_original_data(self, curve_type: str):
        """Update original data for a curve type from the processor."""
        try:
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if not processor:
                return
            
            if curve_type == 'hyperpol':
                if (hasattr(processor, 'modified_hyperpol') and 
                    hasattr(processor, 'modified_hyperpol_times') and
                    processor.modified_hyperpol is not None):
                    self.subtractor.set_original_data(
                        'hyperpol', 
                        processor.modified_hyperpol, 
                        processor.modified_hyperpol_times
                    )
            elif curve_type == 'depol':
                if (hasattr(processor, 'modified_depol') and 
                    hasattr(processor, 'modified_depol_times') and
                    processor.modified_depol is not None):
                    self.subtractor.set_original_data(
                        'depol', 
                        processor.modified_depol, 
                        processor.modified_depol_times
                    )
                    
        except Exception as e:
            logger.error(f"Failed to update original data for {curve_type}: {str(e)}")
    
    def _update_button_states(self):
        """Update button states based on available data."""
        has_hyperpol_data = self.subtractor._has_required_data('hyperpol')
        has_depol_data = self.subtractor._has_required_data('depol')
        
        # Enable buttons if we have any data
        self.panel.set_buttons_enabled(has_hyperpol_data or has_depol_data)
    
    def _on_subtraction_requested(self, curve_type: str):
        """Handle subtraction request for a specific curve."""
        try:
            if not self.subtractor._has_required_data(curve_type):
                self.panel.show_error(f"No linear fit data available for {curve_type}")
                return
            
            # Perform subtraction
            subtracted_data, times = self.subtractor.subtract_linear_fit(curve_type)
            
            # Update the processor with subtracted data
            self._update_processor_with_subtracted_data(curve_type, subtracted_data, times)
            
            # Reload the plot
            self._reload_plot()
            
            # Update display
            self.panel.on_subtraction_completed(curve_type, True, "Subtraction completed successfully")
            
            logger.info(f"Successfully subtracted linear fit from {curve_type}")
            
        except Exception as e:
            logger.error(f"Failed to subtract linear fit from {curve_type}: {str(e)}")
            self.panel.on_subtraction_completed(curve_type, False, str(e))
    
    def _on_both_subtraction_requested(self):
        """Handle subtraction request for both curves."""
        try:
            # Perform subtraction for both curves
            results = self.subtractor.subtract_both_curves()
            
            if not results:
                self.panel.show_error("No curves available for subtraction")
                return
            
            # Update processor with all subtracted data
            for curve_type, (data, times) in results.items():
                self._update_processor_with_subtracted_data(curve_type, data, times)
            
            # Reload the plot
            self._reload_plot()
            
            # Update display
            self.panel.on_both_subtraction_completed(results, True, "Both curves subtracted successfully")
            
            logger.info(f"Successfully subtracted linear fits from: {list(results.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to subtract both curves: {str(e)}")
            self.panel.on_both_subtraction_completed({}, False, str(e))
    
    def _on_reset_requested(self, curve_type: str):
        """Handle reset request."""
        try:
            self.subtractor.reset_subtraction(curve_type)
            
            # If we reset all or the current curve, restore original data
            if curve_type == 'all':
                self._restore_original_data('hyperpol')
                self._restore_original_data('depol')
            else:
                self._restore_original_data(curve_type)
            
            # Reload the plot
            self._reload_plot()
            
            # Update display
            self.panel.on_reset_completed(curve_type, True, "Reset completed successfully")
            
            logger.info(f"Successfully reset {curve_type} subtraction data")
            
        except Exception as e:
            logger.error(f"Failed to reset {curve_type}: {str(e)}")
            self.panel.on_reset_completed(curve_type, False, str(e))
    
    def _on_plot_reload_requested(self):
        """Handle plot reload request."""
        try:
            self._reload_plot()
            self.panel.update_status("Plot reloaded", "green")
            logger.info("Plot reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload plot: {str(e)}")
            self.panel.show_error(f"Failed to reload plot: {str(e)}")
    
    def _update_processor_with_subtracted_data(self, curve_type: str, data: np.ndarray, times: np.ndarray):
        """Update the processor with subtracted data."""
        try:
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if not processor:
                logger.warning("No processor available for updating subtracted data")
                return
            
            if curve_type == 'hyperpol':
                processor.modified_hyperpol = data
                processor.modified_hyperpol_times = times
            elif curve_type == 'depol':
                processor.modified_depol = data
                processor.modified_depol_times = times
            
            logger.info(f"Updated processor with subtracted {curve_type} data")
            
        except Exception as e:
            logger.error(f"Failed to update processor with {curve_type} data: {str(e)}")
            raise
    
    def _restore_original_data(self, curve_type: str):
        """Restore original data to the processor."""
        try:
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if not processor:
                return
            
            original_data = self.subtracted_data[curve_type]['original_data']
            original_times = self.subtracted_data[curve_type]['original_times']
            
            if original_data is not None and original_times is not None:
                if curve_type == 'hyperpol':
                    processor.modified_hyperpol = original_data
                    processor.modified_hyperpol_times = original_times
                elif curve_type == 'depol':
                    processor.modified_depol = original_data
                    processor.modified_depol_times = original_times
                
                logger.info(f"Restored original {curve_type} data to processor")
            
        except Exception as e:
            logger.error(f"Failed to restore original {curve_type} data: {str(e)}")
    
    def _reload_plot(self):
        """Reload the main plot with updated data."""
        try:
            # Try different plot update methods
            update_methods = [
                ('update_plot_with_processed_data', self.main_app),
                ('update_plot', self.main_app),
                ('refresh_plot', self.main_app),
            ]
            
            # Get current processor data
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if not processor:
                logger.warning("No processor available for plot reload")
                return
            
            # Try to get the necessary data for plot update
            processed_data = getattr(processor, 'processed_data', None)
            orange_curve = getattr(processor, 'orange_curve', None)
            orange_times = getattr(processor, 'orange_times', None)
            normalized_curve = getattr(processor, 'normalized_curve', None)
            normalized_times = getattr(processor, 'normalized_curve_times', None)
            average_curve = getattr(processor, 'average_curve', None)
            average_times = getattr(processor, 'average_curve_times', None)
            
            # Try each update method
            for method_name, obj in update_methods:
                if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
                    try:
                        method = getattr(obj, method_name)
                        
                        # Call with appropriate parameters
                        if method_name == 'update_plot_with_processed_data':
                            method(processed_data, orange_curve, orange_times, 
                                  normalized_curve, normalized_times, 
                                  average_curve, average_times,
                                  force_full_range=False,
                                  force_auto_scale=False)
                        else:
                            method()
                        
                        logger.info(f"Successfully reloaded plot using {method_name}")
                        return
                        
                    except Exception as e:
                        logger.debug(f"Plot update method {method_name} failed: {str(e)}")
                        continue
            
            logger.warning("All plot update methods failed")
            
        except Exception as e:
            logger.error(f"Failed to reload plot: {str(e)}")
            raise
    
    def update_from_processor_change(self):
        """Update the integration when processor data changes."""
        try:
            # Update original data for both curves
            self._update_original_data('hyperpol')
            self._update_original_data('depol')
            
            # Update button states
            self._update_button_states()
            
            # Update display
            self.panel.update_display()
            
            logger.info("Updated integration from processor change")
            
        except Exception as e:
            logger.error(f"Failed to update from processor change: {str(e)}")
