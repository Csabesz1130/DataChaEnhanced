"""
Quick Integration Script for Linear Fit Subtraction
==================================================
This script quickly integrates the subtraction feature with the running application.
"""

import logging
from src.analysis.linear_fit_subtractor import LinearFitSubtractor
from src.gui.linear_fit_subtraction_gui import LinearFitSubtractionPanel

logger = logging.getLogger(__name__)

def integrate_subtraction_feature(main_app):
    """
    Quickly integrate the subtraction feature with the running application.
    """
    try:
        # Create the subtractor
        subtractor = LinearFitSubtractor()
        
        # Create the GUI panel
        panel = LinearFitSubtractionPanel()
        panel.set_subtractor(subtractor)
        
        # Store in main app for easy access
        main_app.linear_fit_subtractor = subtractor
        main_app.linear_fit_subtraction_panel = panel
        
        # Connect to curve fitting manager if available
        if hasattr(main_app, 'curve_fitting_panel') and main_app.curve_fitting_panel:
            _enhance_curve_fitting_manager(main_app, subtractor)
        
        # Update processor data
        _update_processor_data(main_app, subtractor)
        
        logger.info("Linear fit subtraction feature integrated successfully")
        return subtractor, panel
        
    except Exception as e:
        logger.error(f"Failed to integrate subtraction feature: {str(e)}")
        raise

def _enhance_curve_fitting_manager(main_app, subtractor):
    """Enhance the curve fitting manager to work with our subtractor."""
    try:
        curve_fitting_panel = main_app.curve_fitting_panel
        
        if hasattr(curve_fitting_panel, 'fitting_manager'):
            fitting_manager = curve_fitting_panel.fitting_manager
            
            # Override the apply_linear_correction method to use our subtractor
            original_apply_correction = fitting_manager.apply_linear_correction
            
            def enhanced_apply_correction(curve_type, operation='subtract'):
                # Call original method
                result = original_apply_correction(curve_type, operation)
                
                if result and operation == 'subtract':
                    # Update our subtractor with the fitted data
                    linear_params = fitting_manager.fitted_curves[curve_type]['linear_params']
                    linear_curve = fitting_manager.fitted_curves[curve_type]['linear_curve']
                    r_squared = fitting_manager.fitted_curves[curve_type]['r_squared_linear']
                    
                    if linear_params and linear_curve and r_squared is not None:
                        # Set fitted curves
                        subtractor.set_fitted_curves(curve_type, linear_params, linear_curve, r_squared)
                        
                        # Set original data
                        data = fitting_manager.curve_data[curve_type]['data']
                        times = fitting_manager.curve_data[curve_type]['times']
                        subtractor.set_original_data(curve_type, data, times)
                        
                        # Perform subtraction
                        subtracted_data, subtracted_times = subtractor.subtract_linear_fit(curve_type)
                        
                        # Update the processor with subtracted data
                        _update_processor_with_subtracted_data(main_app, curve_type, subtracted_data, subtracted_times)
                        
                        # Reload the plot
                        _reload_plot(main_app)
                        
                        logger.info(f"Applied linear subtraction to {curve_type} and updated plot")
                
                return result
            
            # Replace the method
            fitting_manager.apply_linear_correction = enhanced_apply_correction
            
            logger.info("Enhanced curve fitting manager with subtraction feature")
            
    except Exception as e:
        logger.error(f"Failed to enhance curve fitting manager: {str(e)}")

def _update_processor_data(main_app, subtractor):
    """Update the subtractor with current processor data."""
    try:
        processor = getattr(main_app, 'action_potential_processor', None)
        if not processor:
            return
        
        # Update hyperpol data
        if (hasattr(processor, 'modified_hyperpol') and 
            hasattr(processor, 'modified_hyperpol_times') and
            processor.modified_hyperpol is not None):
            subtractor.set_original_data(
                'hyperpol', 
                processor.modified_hyperpol, 
                processor.modified_hyperpol_times
            )
        
        # Update depol data
        if (hasattr(processor, 'modified_depol') and 
            hasattr(processor, 'modified_depol_times') and
            processor.modified_depol is not None):
            subtractor.set_original_data(
                'depol', 
                processor.modified_depol, 
                processor.modified_depol_times
            )
        
        logger.info("Updated subtractor with current processor data")
        
    except Exception as e:
        logger.error(f"Failed to update processor data: {str(e)}")

def _update_processor_with_subtracted_data(main_app, curve_type, data, times):
    """Update the processor with subtracted data."""
    try:
        processor = getattr(main_app, 'action_potential_processor', None)
        if not processor:
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

def _reload_plot(main_app):
    """Reload the main plot."""
    try:
        # Try different plot update methods
        update_methods = [
            'update_plot_with_processed_data',
            'update_plot',
            'refresh_plot'
        ]
        
        for method_name in update_methods:
            if hasattr(main_app, method_name):
                try:
                    method = getattr(main_app, method_name)
                    
                    if method_name == 'update_plot_with_processed_data':
                        # Get processor data
                        processor = getattr(main_app, 'action_potential_processor', None)
                        if processor:
                            processed_data = getattr(processor, 'processed_data', None)
                            orange_curve = getattr(processor, 'orange_curve', None)
                            orange_times = getattr(processor, 'orange_times', None)
                            normalized_curve = getattr(processor, 'normalized_curve', None)
                            normalized_times = getattr(processor, 'normalized_curve_times', None)
                            average_curve = getattr(processor, 'average_curve', None)
                            average_times = getattr(processor, 'average_curve_times', None)
                            
                            method(processed_data, orange_curve, orange_times, 
                                  normalized_curve, normalized_times, 
                                  average_curve, average_times,
                                  force_full_range=False,
                                  force_auto_scale=False)
                        else:
                            method()
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

def apply_subtraction_to_current_fit(main_app):
    """Apply subtraction to the currently fitted curve."""
    try:
        if not hasattr(main_app, 'linear_fit_subtractor'):
            logger.warning("Subtraction feature not integrated yet")
            return
        
        subtractor = main_app.linear_fit_subtractor
        
        # Check if we have hyperpol fit data
        if subtractor._has_required_data('hyperpol'):
            logger.info("Applying subtraction to hyperpol curve...")
            subtracted_data, times = subtractor.subtract_linear_fit('hyperpol')
            _update_processor_with_subtracted_data(main_app, 'hyperpol', subtracted_data, times)
            _reload_plot(main_app)
            logger.info("Hyperpol subtraction applied and plot reloaded")
        
        # Check if we have depol fit data
        if subtractor._has_required_data('depol'):
            logger.info("Applying subtraction to depol curve...")
            subtracted_data, times = subtractor.subtract_linear_fit('depol')
            _update_processor_with_subtracted_data(main_app, 'depol', subtracted_data, times)
            _reload_plot(main_app)
            logger.info("Depol subtraction applied and plot reloaded")
        
    except Exception as e:
        logger.error(f"Failed to apply subtraction to current fit: {str(e)}")
