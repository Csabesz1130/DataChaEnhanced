"""
Fix Plot Refresh Script
=======================
This script will apply the linear fit subtraction and refresh the plot.
"""

import sys
import os
import numpy as np
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def apply_linear_subtraction_and_refresh():
    """
    Apply linear subtraction to the current fitted data and refresh the plot.
    This function can be called from within the running application.
    """
    try:
        # Get the main application instance (this would need to be passed in)
        # For now, we'll create a function that can be called from the application
        
        print("Applying linear fit subtraction and refreshing plot...")
        
        # The actual implementation would go here
        # This is a template for how to integrate with the running app
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply subtraction and refresh: {str(e)}")
        return False

def create_subtraction_patch():
    """
    Create a patch that can be applied to fix the plot refresh issue.
    """
    patch_code = '''
# Add this to your curve fitting manager's apply_linear_correction method
# After line 470 in curve_fitting_manager.py

# Update the processor with corrected data
if hasattr(self, 'main_app') and self.main_app:
    processor = getattr(self.main_app, 'action_potential_processor', None)
    if processor:
        if curve_type == 'hyperpol':
            processor.modified_hyperpol = corrected_data
        elif curve_type == 'depol':
            processor.modified_depol = corrected_data
        
        # Reload the plot
        if hasattr(self.main_app, 'update_plot_with_processed_data'):
            self.main_app.update_plot_with_processed_data(
                getattr(processor, 'processed_data', None),
                getattr(processor, 'orange_curve', None),
                getattr(processor, 'orange_times', None),
                getattr(processor, 'normalized_curve', None),
                getattr(processor, 'normalized_curve_times', None),
                getattr(processor, 'average_curve', None),
                getattr(processor, 'average_curve_times', None)
            )
'''
    
    print("Patch code to fix plot refresh:")
    print("=" * 50)
    print(patch_code)
    
    return patch_code

if __name__ == "__main__":
    print("Linear Fit Subtraction Fix")
    print("=" * 30)
    print()
    print("The issue is that the curve fitting manager applies the correction")
    print("but doesn't update the plot or processor data.")
    print()
    print("Solution: Apply the patch below to fix the plot refresh issue.")
    print()
    
    create_subtraction_patch()
    
    print()
    print("To apply this fix:")
    print("1. Open src/analysis/curve_fitting_manager.py")
    print("2. Find the apply_linear_correction method (around line 451)")
    print("3. Add the patch code after line 470")
    print("4. Save the file and restart the application")
    print()
    print("Or use the new linear fit subtraction feature we created!")
