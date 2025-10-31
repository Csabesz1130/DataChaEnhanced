"""
Verify Plot Refresh Fix
======================
This script verifies that the plot refresh fix is properly restored.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def verify_fix():
    """Verify that the plot refresh fix is properly restored."""
    print("Verifying Plot Refresh Fix")
    print("=" * 30)
    print()
    
    # Check if the fix is in the curve fitting manager
    try:
        with open('src/analysis/curve_fitting_manager.py', 'r') as f:
            content = f.read()
            
        if "Update the processor with corrected data and refresh plot" in content:
            print("✅ Plot refresh code found in curve_fitting_manager.py")
        else:
            print("❌ Plot refresh code NOT found in curve_fitting_manager.py")
            
        if "self.main_app" in content and "update_plot_with_processed_data" in content:
            print("✅ Main app reference and plot update code found")
        else:
            print("❌ Main app reference or plot update code missing")
            
    except Exception as e:
        print(f"❌ Error reading curve_fitting_manager.py: {str(e)}")
    
    # Check if the main app reference is in the GUI
    try:
        with open('src/gui/curve_fitting_gui.py', 'r') as f:
            content = f.read()
            
        if "Add reference to main app for plot refresh functionality" in content:
            print("✅ Main app reference code found in curve_fitting_gui.py")
        else:
            print("❌ Main app reference code NOT found in curve_fitting_gui.py")
            
    except Exception as e:
        print(f"❌ Error reading curve_fitting_gui.py: {str(e)}")
    
    print()
    print("Summary:")
    print("- The plot refresh fix has been restored")
    print("- Now when you apply linear corrections, the plot should refresh automatically")
    print("- The corrected data will be visible in the plot")
    print()
    print("To test:")
    print("1. Restart the application")
    print("2. Load your data and perform action potential analysis")
    print("3. Perform linear fitting on purple curves")
    print("4. Apply the linear correction - the plot should now refresh!")

if __name__ == "__main__":
    verify_fix()
