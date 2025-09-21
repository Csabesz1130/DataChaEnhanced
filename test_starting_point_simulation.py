#!/usr/bin/env python3
"""
Test script for the Starting Point Simulation feature.

This script demonstrates how to use the simulation feature to find the optimal
starting point for ActionPotentialTab analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.starting_point_simulator import StartingPointSimulator
from src.utils.logger import app_logger

def create_test_signal():
    """Create a test signal with known characteristics."""
    # Time data (10 seconds at 1 kHz sampling rate)
    time_data = np.linspace(0, 10, 10000)
    
    # Create a signal with periodic spikes and smooth segments
    signal = np.zeros_like(time_data)
    
    # Add baseline
    signal += -80  # pA baseline
    
    # Add smooth exponential decay segments
    for i in range(5):
        start_time = 1 + i * 1.8
        end_time = start_time + 0.5
        
        # Find indices
        start_idx = int(start_time * 1000)
        end_idx = int(end_time * 1000)
        
        if end_idx < len(signal):
            # Create exponential decay
            t_segment = time_data[start_idx:end_idx] - time_data[start_idx]
            decay = 20 * np.exp(-t_segment * 5)  # 20 pA amplitude, 200ms time constant
            signal[start_idx:end_idx] += decay
    
    # Add some noise
    noise = np.random.normal(0, 2, len(signal))
    signal += noise
    
    # Add periodic spikes at specific intervals (simulating the problem)
    spike_interval = 200  # points
    for i in range(1, len(signal) // spike_interval):
        spike_idx = i * spike_interval
        if spike_idx < len(signal):
            signal[spike_idx] += 50  # Add 50 pA spike
    
    return signal, time_data

def test_simulation():
    """Test the starting point simulation."""
    print("Creating test signal...")
    signal, time_data = create_test_signal()
    
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
    
    print("Running starting point simulation...")
    
    # Create simulator
    simulator = StartingPointSimulator(signal, time_data, params)
    
    # Set simulation parameters
    simulator.start_point_range = (10, 80)
    simulator.step_size = 10
    
    # Progress callback
    def progress_callback(progress, message):
        print(f"\r{message} ({progress:.1f}%)", end="", flush=True)
    
    # Run simulation
    results = simulator.run_simulation(progress_callback)
    
    print("\n\nSimulation Results:")
    print("==================")
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    optimal_point = results['optimal_starting_point']
    recommendation = results['recommendation']
    
    print(f"Optimal Starting Point: {optimal_point}")
    print(f"Confidence: {recommendation['confidence']}")
    print(f"Quality: {recommendation['quality']}")
    print(f"Reason: {recommendation['reason']}")
    print(f"Smoothness Improvement: {recommendation['smoothness_improvement']}")
    print(f"Outstanding Points Reduction: {recommendation['outstanding_points_reduction']}")
    
    print("\nDetailed Results:")
    print("=================")
    
    for start_point, result in sorted(results['results'].items()):
        if 'error' not in result:
            smoothness = result['smoothness_score']
            outstanding = result['outstanding_points']
            quality = result['overall_quality']
            print(f"Point {start_point:2d}: Smoothness={smoothness:.3f}, Outstanding={outstanding:2d}, Quality={quality}")
    
    # Test the optimal starting point
    print(f"\nTesting optimal starting point {optimal_point}...")
    optimal_result = simulator.get_detailed_analysis(optimal_point)
    
    if optimal_result and 'processor' in optimal_result:
        processor = optimal_result['processor']
        
        # Check if purple curves were generated
        if hasattr(processor, 'modified_hyperpol') and hasattr(processor, 'modified_depol'):
            print(f"Purple curves generated successfully:")
            print(f"  Hyperpol points: {len(processor.modified_hyperpol)}")
            print(f"  Depol points: {len(processor.modified_depol)}")
            print(f"  Smoothness score: {optimal_result['smoothness_score']:.3f}")
            print(f"  Outstanding points: {optimal_result['outstanding_points']}")
        else:
            print("Purple curves not generated")
    
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    test_simulation()
