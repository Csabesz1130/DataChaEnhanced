"""
Test Script for Curve Fitting Implementation
============================================
Location: test_curve_fitting.py

Run this to test the curve fitting functionality.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockProcessor:
    """Mock processor to generate test data."""
    
    def __init__(self):
        # Generate realistic purple curve data
        self.modified_hyperpol_times = np.linspace(0, 0.1, 200)  # 0-100ms in seconds
        self.modified_depol_times = np.linspace(0, 0.1, 200)
        
        # Hyperpolarization: starts positive, decays to negative with linear drift
        linear_component = -50 * self.modified_hyperpol_times  # Linear drift
        exp_component = 30 * np.exp(-self.modified_hyperpol_times / 0.02)  # Exponential decay
        noise = np.random.normal(0, 1, len(self.modified_hyperpol_times))
        self.modified_hyperpol = linear_component + exp_component - 20 + noise
        
        # Depolarization: starts negative, rises to less negative with linear drift
        linear_component = 20 * self.modified_depol_times  # Linear drift
        exp_component = -30 * (1 - np.exp(-self.modified_depol_times / 0.025))  # Exponential rise
        noise = np.random.normal(0, 1, len(self.modified_depol_times))
        self.modified_depol = linear_component + exp_component - 25 + noise
        
        # Store test parameters
        self.test_params = {
            'hyperpol': {'slope': -50, 'tau': 0.02, 'A': 30},
            'depol': {'slope': 20, 'tau': 0.025, 'A': -30}
        }

class TestApp:
    """Test application for curve fitting."""
    
    def __init__(self):
        # Create mock processor
        self.action_potential_processor = MockProcessor()
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize curve fitting manager
        from src.analysis.curve_fitting_manager import CurveFittingManager
        self.fitting_manager = CurveFittingManager(self.fig, self.ax)
        
        # Set up data
        self.fitting_manager.set_curve_data(
            'hyperpol',
            self.action_potential_processor.modified_hyperpol,
            self.action_potential_processor.modified_hyperpol_times
        )
        self.fitting_manager.set_curve_data(
            'depol',
            self.action_potential_processor.modified_depol,
            self.action_potential_processor.modified_depol_times
        )
        
        # Plot initial curves
        self.plot_curves()
        
        logger.info("Test app initialized")
    
    def plot_curves(self):
        """Plot the test curves."""
        # Clear axes
        self.ax.clear()
        
        # Plot hyperpol
        self.ax.plot(
            self.action_potential_processor.modified_hyperpol_times * 1000,
            self.action_potential_processor.modified_hyperpol,
            'b-', linewidth=2, label='Hyperpol (Purple)', alpha=0.7
        )
        
        # Plot depol
        self.ax.plot(
            self.action_potential_processor.modified_depol_times * 1000,
            self.action_potential_processor.modified_depol,
            'r-', linewidth=2, label='Depol (Purple)', alpha=0.7
        )
        
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Current (pA)')
        self.ax.set_title('Curve Fitting Test - Click Points for Manual Fitting')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.tight_layout()
    
    def test_automatic_fitting(self):
        """Test automatic fitting with predefined points."""
        logger.info("Testing automatic fitting...")
        
        # Test hyperpol
        data = self.action_potential_processor.modified_hyperpol
        times = self.action_potential_processor.modified_hyperpol_times
        
        # Select points for linear fit (10% to 30%)
        idx1, idx2 = 20, 60
        self.fitting_manager.selected_points['hyperpol']['linear_points'] = [
            {'index': idx1, 'time': times[idx1], 'value': data[idx1]},
            {'index': idx2, 'time': times[idx2], 'value': data[idx2]}
        ]
        self.fitting_manager._fit_linear('hyperpol')
        
        # Select point for exponential fit (50%)
        idx3 = 100
        self.fitting_manager.selected_points['hyperpol']['exp_point'] = {
            'index': idx3, 'time': times[idx3], 'value': data[idx3]
        }
        self.fitting_manager._fit_exponential('hyperpol')
        
        # Test depol
        data = self.action_potential_processor.modified_depol
        times = self.action_potential_processor.modified_depol_times
        
        # Select points for linear fit
        idx1, idx2 = 20, 60
        self.fitting_manager.selected_points['depol']['linear_points'] = [
            {'index': idx1, 'time': times[idx1], 'value': data[idx1]},
            {'index': idx2, 'time': times[idx2], 'value': data[idx2]}
        ]
        self.fitting_manager._fit_linear('depol')
        
        # Select point for exponential fit
        idx3 = 100
        self.fitting_manager.selected_points['depol']['exp_point'] = {
            'index': idx3, 'time': times[idx3], 'value': data[idx3]
        }
        self.fitting_manager._fit_exponential('depol')
        
        # Display results
        results = self.fitting_manager.get_fitting_results()
        self.display_results(results)
        
        return results
    
    def display_results(self, results):
        """Display fitting results."""
        print("\n" + "="*60)
        print("CURVE FITTING TEST RESULTS")
        print("="*60)
        
        for curve_type in ['hyperpol', 'depol']:
            if curve_type in results:
                print(f"\n{curve_type.upper()} CURVE:")
                print("-" * 40)
                
                if 'linear' in results[curve_type]:
                    linear = results[curve_type]['linear']
                    print(f"Linear Fit:")
                    print(f"  {linear['equation']}")
                    print(f"  Expected slope: {self.action_potential_processor.test_params[curve_type]['slope']}")
                    print(f"  Fitted slope: {linear['slope']:.2f}")
                    print(f"  R²: {linear['r_squared']:.4f}")
                
                if 'exponential' in results[curve_type]:
                    exp = results[curve_type]['exponential']
                    print(f"Exponential Fit:")
                    print(f"  Amplitude: {exp['A']:.2f}")
                    print(f"  Time constant: {exp['tau']:.4f}")
                    print(f"  Expected τ: {self.action_potential_processor.test_params[curve_type]['tau']}")
                    print(f"  Model type: {exp['model_type']}")
                    print(f"  R²: {exp['r_squared']:.4f}")
        
        print("\n" + "="*60)
    
    def test_corrections(self):
        """Test linear corrections."""
        logger.info("Testing linear corrections...")
        
        # Apply corrections
        hyperpol_result = self.fitting_manager.apply_linear_correction('hyperpol', 'subtract')
        depol_result = self.fitting_manager.apply_linear_correction('depol', 'add')
        
        if hyperpol_result and depol_result:
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Hyperpol original vs corrected
            axes[0,0].plot(hyperpol_result['times']*1000, hyperpol_result['original'], 
                          'b-', label='Original', alpha=0.7)
            axes[0,0].plot(hyperpol_result['times']*1000, hyperpol_result['trend'], 
                          'g--', label='Linear Trend', alpha=0.7)
            axes[0,0].set_title('Hyperpol - Original & Trend')
            axes[0,0].set_xlabel('Time (ms)')
            axes[0,0].set_ylabel('Current (pA)')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            axes[0,1].plot(hyperpol_result['times']*1000, hyperpol_result['corrected'], 
                          'b-', label='Corrected', alpha=0.7)
            axes[0,1].set_title('Hyperpol - Corrected (Trend Subtracted)')
            axes[0,1].set_xlabel('Time (ms)')
            axes[0,1].set_ylabel('Current (pA)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Depol original vs corrected
            axes[1,0].plot(depol_result['times']*1000, depol_result['original'], 
                          'r-', label='Original', alpha=0.7)
            axes[1,0].plot(depol_result['times']*1000, depol_result['trend'], 
                          'g--', label='Linear Trend', alpha=0.7)
            axes[1,0].set_title('Depol - Original & Trend')
            axes[1,0].set_xlabel('Time (ms)')
            axes[1,0].set_ylabel('Current (pA)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            axes[1,1].plot(depol_result['times']*1000, depol_result['corrected'], 
                          'r-', label='Corrected', alpha=0.7)
            axes[1,1].set_title('Depol - Corrected (Trend Added)')
            axes[1,1].set_xlabel('Time (ms)')
            axes[1,1].set_ylabel('Current (pA)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            fig.suptitle('Linear Trend Corrections')
            fig.tight_layout()
            
            logger.info("Corrections applied successfully")
            return fig
        
        return None
    
    def run_interactive(self):
        """Run interactive test."""
        print("\n" + "="*60)
        print("INTERACTIVE CURVE FITTING TEST")
        print("="*60)
        print("\n1. Click on the plot to select points")
        print("2. First select 2 points for linear fit")
        print("3. Then select 1 point for exponential fit start")
        print("\nPress 'h' to start hyperpol linear fitting")
        print("Press 'd' to start depol linear fitting")
        print("Press 'e' to start exponential fitting")
        print("Press 'c' to clear all fits")
        print("Press 'r' to show results")
        print("Press 'q' to quit")
        
        def on_key(event):
            if event.key == 'h':
                self.fitting_manager.start_linear_selection('hyperpol')
                print("Select 2 points on hyperpol curve for linear fit...")
            elif event.key == 'd':
                self.fitting_manager.start_linear_selection('depol')
                print("Select 2 points on depol curve for linear fit...")
            elif event.key == 'e':
                if self.fitting_manager.fitted_curves['hyperpol']['linear_params']:
                    self.fitting_manager.start_exp_selection('hyperpol')
                    print("Select 1 point on hyperpol curve for exponential fit...")
                elif self.fitting_manager.fitted_curves['depol']['linear_params']:
                    self.fitting_manager.start_exp_selection('depol')
                    print("Select 1 point on depol curve for exponential fit...")
                else:
                    print("Perform linear fitting first!")
            elif event.key == 'c':
                self.fitting_manager.clear_fits()
                print("All fits cleared")
            elif event.key == 'r':
                results = self.fitting_manager.get_fitting_results()
                self.display_results(results)
            elif event.key == 'q':
                plt.close('all')
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

def main():
    """Main test function."""
    print("DataChaEnhanced Curve Fitting Test")
    print("==================================")
    
    try:
        # Create test app
        app = TestApp()
        
        # Run automatic test
        print("\n1. Running automatic fitting test...")
        results = app.test_automatic_fitting()
        
        # Test corrections
        print("\n2. Testing linear corrections...")
        corrections_fig = app.test_corrections()
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        app.fig.savefig(f'test_fitting_{timestamp}.png', dpi=150)
        print(f"\nTest plot saved: test_fitting_{timestamp}.png")
        
        if corrections_fig:
            corrections_fig.savefig(f'test_corrections_{timestamp}.png', dpi=150)
            print(f"Corrections plot saved: test_corrections_{timestamp}.png")
        
        # Ask for interactive test
        response = input("\nRun interactive test? (y/n): ")
        if response.lower() == 'y':
            app.run_interactive()
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)