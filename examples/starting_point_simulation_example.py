#!/usr/bin/env python3
"""
Example: Starting Point Simulation for ActionPotentialTab

This example demonstrates how to use the starting point simulation feature
to find the optimal starting point for action potential analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.starting_point_simulator import StartingPointSimulator
from src.analysis.action_potential import ActionPotentialProcessor
from src.utils.logger import app_logger

def create_realistic_test_signal():
    """
    Create a realistic test signal that mimics action potential data
    with periodic spikes that can be optimized by choosing the right starting point.
    """
    # Time data (5 seconds at 1 kHz sampling rate)
    time_data = np.linspace(0, 5, 5000)
    
    # Create a signal with multiple cycles
    signal = np.zeros_like(time_data)
    
    # Add baseline current
    signal += -80  # pA baseline
    
    # Create 2 cycles of action potential-like responses
    for cycle in range(2):
        cycle_start = 0.5 + cycle * 2.0  # Start each cycle at 0.5s and 2.5s
        
        # Hyperpolarization phase (0.2s duration)
        hyp_start = cycle_start
        hyp_end = cycle_start + 0.2
        hyp_idx_start = int(hyp_start * 1000)
        hyp_idx_end = int(hyp_end * 1000)
        
        if hyp_idx_end < len(signal):
            t_hyp = time_data[hyp_idx_start:hyp_idx_end] - time_data[hyp_idx_start]
            # Exponential decay for hyperpolarization
            hyp_response = -30 * np.exp(-t_hyp * 3)  # -30 pA amplitude
            signal[hyp_idx_start:hyp_idx_end] += hyp_response
        
        # Depolarization phase (0.3s duration)
        dep_start = cycle_start + 0.2
        dep_end = cycle_start + 0.5
        dep_idx_start = int(dep_start * 1000)
        dep_idx_end = int(dep_end * 1000)
        
        if dep_idx_end < len(signal):
            t_dep = time_data[dep_idx_start:dep_idx_end] - time_data[dep_idx_start]
            # Exponential rise and decay for depolarization
            dep_response = 40 * (1 - np.exp(-t_dep * 2)) * np.exp(-t_dep * 1.5)
            signal[dep_idx_start:dep_idx_end] += dep_response
    
    # Add periodic spikes that will be problematic with wrong starting point
    # These spikes occur every 200 points (0.2s intervals)
    spike_interval = 200
    for i in range(1, len(signal) // spike_interval):
        spike_idx = i * spike_interval
        if spike_idx < len(signal):
            # Add spikes of varying amplitude
            spike_amplitude = 20 + 10 * np.sin(i * 0.5)  # Varying amplitude
            signal[spike_idx] += spike_amplitude
    
    # Add realistic noise
    noise = np.random.normal(0, 1.5, len(signal))
    signal += noise
    
    return signal, time_data

def run_simulation_example():
    """Run a complete simulation example."""
    print("Starting Point Simulation Example")
    print("=" * 40)
    
    # Create test signal
    print("1. Creating realistic test signal...")
    signal, time_data = create_realistic_test_signal()
    print(f"   Signal length: {len(signal)} points")
    print(f"   Time range: {time_data[0]:.2f} - {time_data[-1]:.2f} seconds")
    
    # Analysis parameters
    params = {
        'n_cycles': 2,
        't0': 20.0,
        't1': 100.0,
        't2': 100.0,
        'V0': -80.0,
        'V1': -100.0,
        'V2': -20.0,
        'use_alternative_method': False
    }
    
    print("\n2. Analysis parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Create simulator
    print("\n3. Creating simulator...")
    simulator = StartingPointSimulator(signal, time_data, params)
    
    # Configure simulation
    simulator.start_point_range = (5, 50)  # Test range 5-50
    simulator.step_size = 5  # Test every 5th point
    
    print(f"   Testing range: {simulator.start_point_range}")
    print(f"   Step size: {simulator.step_size}")
    
    # Progress callback
    def progress_callback(progress, message):
        print(f"\r   {message} ({progress:.1f}%)", end="", flush=True)
    
    # Run simulation
    print("\n4. Running simulation...")
    results = simulator.run_simulation(progress_callback)
    
    print("\n\n5. Simulation Results:")
    print("=" * 25)
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    # Display results
    optimal_point = results['optimal_starting_point']
    recommendation = results['recommendation']
    
    print(f"Optimal Starting Point: {optimal_point}")
    print(f"Confidence Level: {recommendation['confidence'].upper()}")
    print(f"Quality Assessment: {recommendation['quality'].upper()}")
    print(f"Reason: {recommendation['reason']}")
    print(f"Smoothness Improvement: {recommendation['smoothness_improvement']}")
    print(f"Outstanding Points Reduction: {recommendation['outstanding_points_reduction']}")
    
    # Show detailed results
    print("\n6. Detailed Results by Starting Point:")
    print("-" * 45)
    print("Point | Smoothness | Outstanding | Quality")
    print("-" * 45)
    
    for start_point, result in sorted(results['results'].items()):
        if 'error' not in result:
            smoothness = result['smoothness_score']
            outstanding = result['outstanding_points']
            quality = result['overall_quality']
            print(f"{start_point:5d} | {smoothness:10.3f} | {outstanding:11d} | {quality}")
    
    # Test the optimal starting point
    print(f"\n7. Testing optimal starting point {optimal_point}...")
    optimal_result = simulator.get_detailed_analysis(optimal_point)
    
    if optimal_result and 'processor' in optimal_result:
        processor = optimal_result['processor']
        
        # Check purple curves
        if (hasattr(processor, 'modified_hyperpol') and 
            hasattr(processor, 'modified_depol') and
            processor.modified_hyperpol is not None and
            processor.modified_depol is not None):
            
            print(f"   Purple curves generated successfully:")
            print(f"   - Hyperpol points: {len(processor.modified_hyperpol)}")
            print(f"   - Depol points: {len(processor.modified_depol)}")
            print(f"   - Smoothness score: {optimal_result['smoothness_score']:.3f}")
            print(f"   - Outstanding points: {optimal_result['outstanding_points']}")
            
            # Show curve statistics
            hyp_metrics = optimal_result.get('hyperpol_metrics', {})
            dep_metrics = optimal_result.get('depol_metrics', {})
            
            if hyp_metrics:
                print(f"   - Hyperpol variance: {hyp_metrics.get('variance', 0):.2f}")
                print(f"   - Hyperpol skewness: {hyp_metrics.get('skewness', 0):.3f}")
            
            if dep_metrics:
                print(f"   - Depol variance: {dep_metrics.get('variance', 0):.2f}")
                print(f"   - Depol skewness: {dep_metrics.get('skewness', 0):.3f}")
        else:
            print("   ERROR: Purple curves not generated")
    
    # Compare with default starting point
    print(f"\n8. Comparison with default starting point (35):")
    default_result = results['results'].get(35, {})
    
    if 'error' not in default_result:
        print(f"   Default (35): Smoothness={default_result['smoothness_score']:.3f}, "
              f"Outstanding={default_result['outstanding_points']}")
        print(f"   Optimal ({optimal_point}): Smoothness={optimal_result['smoothness_score']:.3f}, "
              f"Outstanding={optimal_result['outstanding_points']}")
        
        improvement = (optimal_result['smoothness_score'] - default_result['smoothness_score']) * 100
        print(f"   Improvement: {improvement:+.1f}% smoothness")
    else:
        print("   Default starting point test failed")
    
    print("\n9. Simulation completed successfully!")
    print("   You can now use starting point", optimal_point, "in your ActionPotentialTab analysis.")

def create_visualization():
    """Create a visualization showing the effect of different starting points."""
    print("\n10. Creating visualization...")
    
    # This would create plots showing the difference between
    # optimal and suboptimal starting points
    print("   (Visualization code would go here)")
    print("   - Plot showing signal with spikes")
    print("   - Comparison of purple curves with different starting points")
    print("   - Smoothness metrics visualization")

if __name__ == "__main__":
    try:
        run_simulation_example()
        create_visualization()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
