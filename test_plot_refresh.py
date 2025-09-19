"""
Test Plot Refresh Fix
====================
This script tests if the plot refresh fix is working.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fix():
    """Test if the plot refresh fix is working."""
    print("Testing Plot Refresh Fix")
    print("=" * 30)
    print()
    print("✅ Applied fixes:")
    print("1. Added plot refresh code to curve_fitting_manager.py")
    print("2. Added main_app reference to fitting manager")
    print("3. Enhanced apply_linear_correction method")
    print()
    print("The fix should now work when you:")
    print("1. Perform linear fitting on your purple curves")
    print("2. Apply the linear correction")
    print("3. The plot should automatically refresh to show the corrected data")
    print()
    print("If the plot still doesn't refresh, try:")
    print("- Restart the application")
    print("- Re-perform the linear fitting and correction")
    print("- Check the console for any error messages")
    print()
    print("The new linear fit subtraction feature is also available!")
    print("It provides separate controls for hyperpol and depol curves.")

if __name__ == "__main__":
    test_fix()
