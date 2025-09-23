#!/usr/bin/env python3
"""
Test script to verify data access in ActionPotentialTab simulation.

This script tests the data access methods used in the simulation feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gui.app import SignalAnalyzerApp
from src.gui.action_potential_tab import ActionPotentialTab
import tkinter as tk
import numpy as np

def test_data_access():
    """Test data access methods."""
    print("Testing data access methods...")
    
    # Create a minimal tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        # Create the main app
        app = SignalAnalyzerApp(root)
        
        # Create some test data
        time_data = np.linspace(0, 1, 1000)
        test_data = np.random.randn(1000) * 100
        
        # Set the data using properties
        app.time_data = time_data
        app.data = test_data
        app.filtered_data = test_data.copy()
        
        print(f"Data set successfully:")
        print(f"  time_data: {app.time_data is not None} (length: {len(app.time_data) if app.time_data is not None else 0})")
        print(f"  data: {app.data is not None} (length: {len(app.data) if app.data is not None else 0})")
        print(f"  filtered_data: {app.filtered_data is not None} (length: {len(app.filtered_data) if app.filtered_data is not None else 0})")
        
        # Test the data access methods used in simulation
        print("\nTesting simulation data access:")
        
        # Test property access
        if hasattr(app, 'filtered_data') and app.filtered_data is not None:
            print("✓ Property access works for filtered_data")
        else:
            print("✗ Property access failed for filtered_data")
            
        if hasattr(app, 'time_data') and app.time_data is not None:
            print("✓ Property access works for time_data")
        else:
            print("✗ Property access failed for time_data")
        
        # Test private attribute access
        if hasattr(app, '_filtered_data') and app._filtered_data is not None:
            print("✓ Private attribute access works for filtered_data")
        else:
            print("✗ Private attribute access failed for filtered_data")
            
        if hasattr(app, '_time_data') and app._time_data is not None:
            print("✓ Private attribute access works for time_data")
        else:
            print("✗ Private attribute access failed for time_data")
        
        # Test the actual simulation method
        print("\nTesting simulation method:")
        try:
            # Create a mock parent for ActionPotentialTab
            class MockParent:
                def __init__(self, master):
                    self.master = master
            
            mock_parent = MockParent(app)
            
            # Create ActionPotentialTab instance
            ap_tab = ActionPotentialTab(mock_parent, lambda x: None)
            
            # Test the data access logic from run_starting_point_simulation
            filtered_data = None
            time_data = None
            
            # Try property access first
            if hasattr(app, 'filtered_data') and app.filtered_data is not None:
                filtered_data = app.filtered_data
                print("✓ Using filtered_data from property")
            elif hasattr(app, '_filtered_data') and app._filtered_data is not None:
                filtered_data = app._filtered_data
                print("✓ Using filtered_data from private attribute")
            elif hasattr(app, 'data') and app.data is not None:
                filtered_data = app.data
                print("✓ Using raw data as filtered_data")
            
            if hasattr(app, 'time_data') and app.time_data is not None:
                time_data = app.time_data
                print("✓ Using time_data from property")
            elif hasattr(app, '_time_data') and app._time_data is not None:
                time_data = app._time_data
                print("✓ Using time_data from private attribute")
            
            if filtered_data is not None and time_data is not None:
                print("✓ Simulation data access successful!")
                print(f"  Retrieved data lengths: filtered={len(filtered_data)}, time={len(time_data)}")
            else:
                print("✗ Simulation data access failed")
                print(f"  filtered_data: {filtered_data is not None}")
                print(f"  time_data: {time_data is not None}")
                
        except Exception as e:
            print(f"✗ Error testing simulation method: {str(e)}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        root.destroy()

if __name__ == "__main__":
    test_data_access()
