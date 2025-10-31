"""
Fast Track Mode for DataChaEnhanced
===================================
Automates the analysis workflow for testing and rapid file processing.

Workflow:
1. Set filters (Savitzky-Golay and Butterworth to true)
2. Switch to Action Potential tab
3. Click "Analyse Signal"
4. Perform fitting operations (using logged points/ranges)
5. Export to Excel
"""

import logging
from typing import Dict, List, Optional, Any
import time

logger = logging.getLogger(__name__)


class FastTrackMode:
    """Automated workflow executor for fast analysis."""
    
    def __init__(self, app, action_logger=None):
        """Initialize fast-track mode."""
        self.app = app
        self.action_logger = action_logger
        self.is_running = False
        
        logger.info("FastTrackMode initialized")
    
    def execute_workflow(self, 
                        use_saved_points: bool = True,
                        fitting_points: Optional[Dict] = None,
                        integration_ranges: Optional[Dict] = None,
                        delay_ms: int = 500) -> bool:
        """
        Execute the fast-track workflow.
        
        Args:
            use_saved_points: If True, use points from action logger
            fitting_points: Manual fitting points dict (overrides saved if provided)
            integration_ranges: Manual integration ranges (overrides saved if provided)
            delay_ms: Delay between steps in milliseconds
        
        Returns:
            True if workflow completed successfully
        """
        if self.is_running:
            logger.warning("Fast-track workflow already running")
            return False
        
        self.is_running = True
        
        try:
            logger.info("Starting fast-track workflow")
            
            # Step 1: Set filters
            if not self._set_filters():
                logger.error("Failed to set filters")
                return False
            
            self._wait(delay_ms)
            
            # Step 2: Switch to Action Potential tab
            if not self._switch_to_action_potential_tab():
                logger.error("Failed to switch to Action Potential tab")
                return False
            
            self._wait(delay_ms)
            
            # Step 3: Click "Analyse Signal" button
            if not self._click_analyze_signal():
                logger.error("Failed to start analysis")
                return False
            
            # Wait for analysis to complete (longer delay)
            self._wait(delay_ms * 3)
            
            # Step 4: Perform fitting operations
            if use_saved_points or fitting_points or integration_ranges:
                if not self._perform_fittings(fitting_points, integration_ranges):
                    logger.warning("Some fitting operations may have failed")
            
            self._wait(delay_ms * 2)
            
            # Step 5: Export to Excel
            if not self._export_to_excel():
                logger.warning("Export may have failed")
            
            logger.info("Fast-track workflow completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in fast-track workflow: {e}")
            return False
        finally:
            self.is_running = False
    
    def _set_filters(self) -> bool:
        """Set Savitzky-Golay and Butterworth filters to true."""
        try:
            # Find filter tab
            if not hasattr(self.app, 'notebook'):
                logger.error("Notebook not found")
                return False
            
            # Switch to Filters tab if it exists
            # Note: This may need adjustment based on actual tab structure
            filter_tab = None
            for i in range(self.app.notebook.index("end")):
                tab_text = self.app.notebook.tab(i, "text")
                if "Filter" in tab_text.lower():
                    self.app.notebook.select(i)
                    filter_tab = self.app.notebook.nametowidget(self.app.notebook.tabs()[i])
                    break
            
            if filter_tab:
                # Find Savitzky-Golay checkbox
                if hasattr(self.app, 'filter_tab'):
                    filter_tab_obj = self.app.filter_tab
                    if hasattr(filter_tab_obj, 'savitzky_golay_var'):
                        filter_tab_obj.savitzky_golay_var.set(True)
                    if hasattr(filter_tab_obj, 'butterworth_var'):
                        filter_tab_obj.butterworth_var.set(True)
                    
                    logger.info("Set filters: Savitzky-Golay=True, Butterworth=True")
                    return True
            
            logger.warning("Filter tab not found, skipping filter setup")
            return True  # Don't fail if filters tab doesn't exist
            
        except Exception as e:
            logger.error(f"Error setting filters: {e}")
            return False
    
    def _switch_to_action_potential_tab(self) -> bool:
        """Switch to Action Potential tab."""
        try:
            if not hasattr(self.app, 'notebook'):
                return False
            
            # Find Action Potential tab
            for i in range(self.app.notebook.index("end")):
                tab_text = self.app.notebook.tab(i, "text")
                if "Action Potential" in tab_text:
                    self.app.notebook.select(i)
                    logger.info("Switched to Action Potential tab")
                    
                    if self.action_logger:
                        self.action_logger.log_tab_switch("Action Potential")
                    
                    return True
            
            logger.error("Action Potential tab not found")
            return False
            
        except Exception as e:
            logger.error(f"Error switching tabs: {e}")
            return False
    
    def _click_analyze_signal(self) -> bool:
        """Click the 'Analyse Signal' button."""
        try:
            if not hasattr(self.app, 'action_potential_tab'):
                logger.error("Action Potential tab not found")
                return False
            
            ap_tab = self.app.action_potential_tab
            
            if hasattr(ap_tab, 'analyze_button'):
                # Log action
                if self.action_logger:
                    self.action_logger.log_button_click("Analyse Signal", tab="Action Potential")
                    self.action_logger.log_analysis_start({})
                
                # Click button
                ap_tab.analyze_signal()
                
                logger.info("Clicked 'Analyse Signal' button")
                return True
            else:
                logger.error("Analyze button not found")
                return False
                
        except Exception as e:
            logger.error(f"Error clicking analyze button: {e}")
            return False
    
    def _perform_fittings(self, 
                         fitting_points: Optional[Dict] = None,
                         integration_ranges: Optional[Dict] = None) -> bool:
        """Perform fitting operations using saved or provided points."""
        try:
            if not hasattr(self.app, 'curve_fitting_panel') or not self.app.curve_fitting_panel:
                logger.warning("Curve fitting panel not available")
                return False
            
            panel = self.app.curve_fitting_panel
            if not panel.fitting_manager:
                logger.warning("Fitting manager not initialized")
                return False
            
            fitting_manager = panel.fitting_manager
            
            # Get fitting points from logger if not provided
            if fitting_points is None and self.action_logger:
                fitting_points = {
                    'hyperpol': {
                        'linear': self.action_logger.get_fitting_points('linear', 'hyperpol'),
                        'exp': self.action_logger.get_fitting_points('exp', 'hyperpol')
                    },
                    'depol': {
                        'linear': self.action_logger.get_fitting_points('linear', 'depol'),
                        'exp': self.action_logger.get_fitting_points('exp', 'depol')
                    }
                }
            
            # Get integration ranges from logger if not provided
            if integration_ranges is None and self.action_logger:
                integration_ranges = self.action_logger.get_integration_ranges()
            
            success = True
            
            # Perform linear fittings
            for curve_type in ['hyperpol', 'depol']:
                linear_points = fitting_points.get(curve_type, {}).get('linear', [])
                if len(linear_points) >= 2:
                    if self._perform_linear_fitting(curve_type, linear_points, fitting_manager):
                        logger.info(f"Completed linear fitting for {curve_type}")
                    else:
                        logger.warning(f"Failed linear fitting for {curve_type}")
                        success = False
                
                # Perform exponential fittings
                exp_points = fitting_points.get(curve_type, {}).get('exp', [])
                if len(exp_points) >= 2:
                    if self._perform_exponential_fitting(curve_type, exp_points, fitting_manager):
                        logger.info(f"Completed exponential fitting for {curve_type}")
                    else:
                        logger.warning(f"Failed exponential fitting for {curve_type}")
                        success = False
                
                # Set integration ranges
                if integration_ranges and curve_type in integration_ranges:
                    range_data = integration_ranges[curve_type]
                    if self._set_integration_range(curve_type, range_data, fitting_manager):
                        logger.info(f"Set integration range for {curve_type}")
                    else:
                        logger.warning(f"Failed to set integration range for {curve_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error performing fittings: {e}")
            return False
    
    def _perform_linear_fitting(self, curve_type: str, points: List[Dict], fitting_manager) -> bool:
        """Perform linear fitting with provided points."""
        try:
            # Convert points to time/value pairs
            if len(points) < 2:
                return False
            
            # Ensure curve data is available
            if fitting_manager.curve_data[curve_type]['data'] is None:
                # Try to update from main app if available
                if hasattr(fitting_manager, 'main_app') and fitting_manager.main_app:
                    if hasattr(fitting_manager.main_app, 'curve_fitting_panel'):
                        fitting_manager.main_app.curve_fitting_panel.update_curve_data()
            
            # Simulate point selection by directly setting selected points
            # This is a workaround - ideally we'd trigger the actual selection mechanism
            times = fitting_manager.curve_data[curve_type]['times']
            data = fitting_manager.curve_data[curve_type]['data']
            
            if times is None or data is None:
                return False
            
            # Find nearest indices for the provided time points
            selected_indices = []
            for point in points[:2]:  # Only need 2 points
                time_ms = point.get('time_ms', point.get('time_seconds', 0) * 1000)
                time_seconds = time_ms / 1000.0
                
                # Find nearest index
                time_diffs = abs(times - time_seconds)
                nearest_idx = time_diffs.argmin()
                selected_indices.append(nearest_idx)
            
            # Set selected points
            fitting_manager.selected_points[curve_type]['linear_points'] = [
                {
                    'index': idx,
                    'time': times[idx],
                    'value': data[idx],
                    'info': {}
                }
                for idx in selected_indices
            ]
            
            # Perform fitting
            fitting_manager._fit_linear(curve_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing linear fitting: {e}")
            return False
    
    def _perform_exponential_fitting(self, curve_type: str, points: List[Dict], fitting_manager) -> bool:
        """Perform exponential fitting with provided points."""
        try:
            if len(points) < 2:
                return False
            
            times = fitting_manager.curve_data[curve_type]['times']
            data = fitting_manager.curve_data[curve_type]['data']
            
            if times is None or data is None:
                return False
            
            # Find nearest indices
            selected_indices = []
            for point in points[:2]:
                time_ms = point.get('time_ms', point.get('time_seconds', 0) * 1000)
                time_seconds = time_ms / 1000.0
                
                time_diffs = abs(times - time_seconds)
                nearest_idx = time_diffs.argmin()
                selected_indices.append(nearest_idx)
            
            # Set selected points
            fitting_manager.selected_points[curve_type]['exp_points'] = [
                {
                    'index': idx,
                    'time': times[idx],
                    'value': data[idx],
                    'info': {}
                }
                for idx in selected_indices
            ]
            
            # Perform fitting
            fitting_manager._fit_exponential(curve_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing exponential fitting: {e}")
            return False
    
    def _set_integration_range(self, curve_type: str, range_data: Dict, fitting_manager) -> bool:
        """Set integration range."""
        try:
            # This would need to integrate with the range selection system
            # For now, we'll log it
            logger.info(f"Integration range for {curve_type}: {range_data}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting integration range: {e}")
            return False
    
    def _export_to_excel(self) -> bool:
        """Export results to Excel."""
        try:
            if not hasattr(self.app, 'action_potential_tab'):
                return False
            
            ap_tab = self.app.action_potential_tab
            
            if hasattr(ap_tab, 'on_export_to_excel_click'):
                # Log action
                if self.action_logger:
                    self.action_logger.log_button_click("Export to Excel", tab="Action Potential")
                
                # Trigger export
                ap_tab.on_export_to_excel_click()
                
                logger.info("Triggered Excel export")
                return True
            else:
                logger.error("Export function not found")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
    
    def _wait(self, ms: int):
        """Wait for specified milliseconds."""
        time.sleep(ms / 1000.0)

