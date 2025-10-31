"""
Linear Fit Subtraction Usage Example
===================================
Location: examples/linear_fit_subtraction_usage.py

This example shows how to use the linear fit subtraction feature.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.linear_fit_subtractor import LinearFitSubtractor
from src.gui.linear_fit_subtraction_gui import LinearFitSubtractionPanel
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

def create_sample_data():
    """Create sample purple curve data for demonstration."""
    # Create time axis
    times = np.linspace(0, 1, 1000)  # 1 second, 1000 points
    
    # Create sample hyperpol data with linear drift
    hyperpol_base = 100 * np.exp(-times * 2)  # Exponential decay
    hyperpol_drift = 50 * times  # Linear drift
    hyperpol_data = hyperpol_base + hyperpol_drift + np.random.normal(0, 2, len(times))
    
    # Create sample depol data with linear drift
    depol_base = 80 * (1 - np.exp(-times * 3))  # Exponential rise
    depol_drift = -30 * times  # Linear drift
    depol_data = depol_base + depol_drift + np.random.normal(0, 1.5, len(times))
    
    return times, hyperpol_data, depol_data

def demonstrate_subtraction():
    """Demonstrate the linear fit subtraction functionality."""
    print("Linear Fit Subtraction Demo")
    print("=" * 40)
    
    # Create sample data
    times, hyperpol_data, depol_data = create_sample_data()
    
    # Create subtractor
    subtractor = LinearFitSubtractor()
    
    # Set original data
    subtractor.set_original_data('hyperpol', hyperpol_data, times)
    subtractor.set_original_data('depol', depol_data, times)
    
    # Simulate linear fits (in real usage, these would come from curve fitting manager)
    # Hyperpol linear fit
    hyperpol_params = {
        'slope': 50.0,  # pA/s
        'intercept': 100.0,  # pA
        'start_idx': 0,
        'end_idx': 999,
        'start_time': 0.0,
        'end_time': 1.0
    }
    hyperpol_linear_curve = {
        'times': times,
        'data': 50.0 * times + 100.0
    }
    hyperpol_r_squared = 0.95
    
    # Depol linear fit
    depol_params = {
        'slope': -30.0,  # pA/s
        'intercept': 80.0,  # pA
        'start_idx': 0,
        'end_idx': 999,
        'start_time': 0.0,
        'end_time': 1.0
    }
    depol_linear_curve = {
        'times': times,
        'data': -30.0 * times + 80.0
    }
    depol_r_squared = 0.92
    
    # Set fitted curves
    subtractor.set_fitted_curves('hyperpol', hyperpol_params, hyperpol_linear_curve, hyperpol_r_squared)
    subtractor.set_fitted_curves('depol', depol_params, depol_linear_curve, depol_r_squared)
    
    print("Original data loaded and linear fits set.")
    
    # Perform subtraction
    print("\nPerforming subtraction...")
    results = subtractor.subtract_both_curves()
    
    print(f"Subtraction completed for: {list(results.keys())}")
    
    # Display results
    for curve_type, (data, times) in results.items():
        print(f"\n{curve_type.upper()} Results:")
        print(f"  Original mean: {np.mean(subtractor.subtracted_data[curve_type]['original_data']):.2f} pA")
        print(f"  Subtracted mean: {np.mean(data):.2f} pA")
        print(f"  Mean change: {np.mean(data) - np.mean(subtractor.subtracted_data[curve_type]['original_data']):.2f} pA")
    
    # Create visualization
    create_visualization(times, hyperpol_data, depol_data, results, subtractor)
    
    return subtractor, results

def create_visualization(times, original_hyperpol, original_depol, results, subtractor):
    """Create a visualization of the subtraction results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Linear Fit Subtraction Results', fontsize=16)
    
    # Hyperpol plots
    ax1, ax2 = axes[0, 0], axes[0, 1]
    
    # Original hyperpol
    ax1.plot(times, original_hyperpol, 'purple', alpha=0.7, label='Original')
    if 'hyperpol' in results:
        ax1.plot(times, results['hyperpol'][0], 'darkviolet', linewidth=2, label='Subtracted')
    ax1.set_title('Hyperpol Curve')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current (pA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hyperpol difference
    if 'hyperpol' in results:
        difference = results['hyperpol'][0] - original_hyperpol
        ax2.plot(times, difference, 'red', linewidth=2, label='Subtracted - Original')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Hyperpol: Subtracted - Original')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Current Difference (pA)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Depol plots
    ax3, ax4 = axes[1, 0], axes[1, 1]
    
    # Original depol
    ax3.plot(times, original_depol, 'purple', alpha=0.7, label='Original')
    if 'depol' in results:
        ax3.plot(times, results['depol'][0], 'darkviolet', linewidth=2, label='Subtracted')
    ax3.set_title('Depol Curve')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current (pA)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Depol difference
    if 'depol' in results:
        difference = results['depol'][0] - original_depol
        ax4.plot(times, difference, 'red', linewidth=2, label='Subtracted - Original')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Depol: Subtracted - Original')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Current Difference (pA)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_gui_example():
    """Create a simple GUI example."""
    app = QApplication([])
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Linear Fit Subtraction Example")
    window.setGeometry(100, 100, 600, 400)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    # Create layout
    layout = QVBoxLayout(central_widget)
    
    # Create subtractor and panel
    subtractor = LinearFitSubtractor()
    panel = LinearFitSubtractionPanel()
    panel.set_subtractor(subtractor)
    
    # Add panel to layout
    layout.addWidget(panel)
    
    # Show window
    window.show()
    
    return app, window, subtractor, panel

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Command line demo")
    print("2. GUI demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Command line demo
        subtractor, results = demonstrate_subtraction()
        print("\nDemo completed. Check the plot window for visualization.")
        
    elif choice == "2":
        # GUI demo
        app, window, subtractor, panel = create_gui_example()
        
        # Set up some sample data
        times, hyperpol_data, depol_data = create_sample_data()
        subtractor.set_original_data('hyperpol', hyperpol_data, times)
        subtractor.set_original_data('depol', depol_data, times)
        
        # Set sample linear fits
        hyperpol_params = {'slope': 50.0, 'intercept': 100.0, 'start_idx': 0, 'end_idx': 999, 'start_time': 0.0, 'end_time': 1.0}
        hyperpol_linear_curve = {'times': times, 'data': 50.0 * times + 100.0}
        subtractor.set_fitted_curves('hyperpol', hyperpol_params, hyperpol_linear_curve, 0.95)
        
        depol_params = {'slope': -30.0, 'intercept': 80.0, 'start_idx': 0, 'end_idx': 999, 'start_time': 0.0, 'end_time': 1.0}
        depol_linear_curve = {'times': times, 'data': -30.0 * times + 80.0}
        subtractor.set_fitted_curves('depol', depol_params, depol_linear_curve, 0.92)
        
        panel.update_display()
        
        print("GUI demo started. Use the buttons to test the subtraction feature.")
        app.exec()
        
    else:
        print("Invalid choice. Exiting.")
