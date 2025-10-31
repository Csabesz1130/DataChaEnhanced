"""
Example Integration of Linear Fit Subtraction Feature
====================================================
Location: src/gui/linear_fit_subtraction_example.py

This module shows how to integrate the linear fit subtraction feature
into the main application.
"""

import logging
from src.gui.linear_fit_subtraction_integration import LinearFitSubtractionIntegration

logger = logging.getLogger(__name__)

def integrate_linear_fit_subtraction(main_app):
    """
    Integrate the linear fit subtraction feature into the main application.
    
    Args:
        main_app: Reference to the main application instance
        
    Returns:
        LinearFitSubtractionIntegration: The integration instance
    """
    try:
        # Create the integration instance
        integration = LinearFitSubtractionIntegration(main_app)
        
        # Store reference in main app for easy access
        main_app.linear_fit_subtraction = integration
        
        # Connect to curve fitting manager if available
        if hasattr(main_app, 'curve_fitting_panel') and main_app.curve_fitting_panel:
            _connect_to_curve_fitting(main_app, integration)
        
        # Connect to processor changes
        if hasattr(main_app, 'action_potential_processor'):
            _connect_to_processor_changes(main_app, integration)
        
        logger.info("Linear fit subtraction feature integrated successfully")
        return integration
        
    except Exception as e:
        logger.error(f"Failed to integrate linear fit subtraction feature: {str(e)}")
        raise

def _connect_to_curve_fitting(main_app, integration):
    """Connect the integration to the curve fitting manager."""
    try:
        curve_fitting_panel = main_app.curve_fitting_panel
        
        # Override the fitting completion callback to include subtraction updates
        original_callback = getattr(curve_fitting_panel, 'on_fitting_completed', None)
        
        def enhanced_fitting_callback(curve_type, fit_type, success, params=None):
            # Call original callback if it exists
            if original_callback:
                original_callback(curve_type, fit_type, success, params)
            
            # Update subtraction integration if linear fit was completed
            if success and fit_type == 'linear' and params:
                try:
                    integration.update_from_curve_fitting(
                        curve_type,
                        params.get('linear_params', {}),
                        params.get('linear_curve', {}),
                        params.get('r_squared', 0.0)
                    )
                except Exception as e:
                    logger.error(f"Failed to update subtraction from curve fitting: {str(e)}")
        
        # Replace the callback
        curve_fitting_panel.on_fitting_completed = enhanced_fitting_callback
        
        logger.info("Connected linear fit subtraction to curve fitting manager")
        
    except Exception as e:
        logger.error(f"Failed to connect to curve fitting manager: {str(e)}")

def _connect_to_processor_changes(main_app, integration):
    """Connect the integration to processor changes."""
    try:
        # Override the processor update method to include subtraction updates
        original_update = getattr(main_app, 'update_curve_fitting_after_processor_change', None)
        
        def enhanced_processor_update():
            # Call original update if it exists
            if original_update:
                original_update()
            
            # Update subtraction integration
            try:
                integration.update_from_processor_change()
            except Exception as e:
                logger.error(f"Failed to update subtraction from processor change: {str(e)}")
        
        # Replace the update method
        main_app.update_curve_fitting_after_processor_change = enhanced_processor_update
        
        logger.info("Connected linear fit subtraction to processor changes")
        
    except Exception as e:
        logger.error(f"Failed to connect to processor changes: {str(e)}")

def add_subtraction_panel_to_tab(main_app, tab_widget):
    """
    Add the subtraction panel to a tab widget.
    
    Args:
        main_app: Reference to the main application instance
        tab_widget: The tab widget to add the panel to
    """
    try:
        if not hasattr(main_app, 'linear_fit_subtraction'):
            logger.warning("Linear fit subtraction not integrated yet")
            return
        
        integration = main_app.linear_fit_subtraction
        panel = integration.get_panel()
        
        # Add the panel to the tab widget
        tab_widget.add(panel, text="Linear Fit Subtraction")
        
        logger.info("Added linear fit subtraction panel to tab widget")
        
    except Exception as e:
        logger.error(f"Failed to add subtraction panel to tab: {str(e)}")

def get_subtraction_panel(main_app):
    """
    Get the subtraction panel for manual integration.
    
    Args:
        main_app: Reference to the main application instance
        
    Returns:
        LinearFitSubtractionPanel: The subtraction panel or None if not available
    """
    try:
        if hasattr(main_app, 'linear_fit_subtraction'):
            return main_app.linear_fit_subtraction.get_panel()
        return None
    except Exception as e:
        logger.error(f"Failed to get subtraction panel: {str(e)}")
        return None
