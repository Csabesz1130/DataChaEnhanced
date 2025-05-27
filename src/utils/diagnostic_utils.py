"""Diagnostic utilities for validating and debugging signal processing."""

import numpy as np
from src.utils.logger import app_logger

class DiagnosticUtils:
   """Collection of diagnostic tools for signal processing validation."""
   
   @staticmethod
   def validate_segment_lengths(processor):
       """Validate all segment lengths are exactly 200 points."""
       segments = {
           'normalized': len(processor.normalized_curve) if hasattr(processor, 'normalized_curve') else 0,
           'average': len(processor.average_curve) if hasattr(processor, 'average_curve') else 0,
           'hyperpol': len(processor.modified_hyperpol) if hasattr(processor, 'modified_hyperpol') else 0,
           'depol': len(processor.modified_depol) if hasattr(processor, 'modified_depol') else 0
       }
       
       for name, length in segments.items():
           if length > 0 and length != 200:
               app_logger.error(f"{name} segment length is {length}, expected 200")
               return False
       return True

   @staticmethod 
   def check_value_ranges(processor):
       """Check value ranges of all curves for anomalies."""
       curves = {
           'orange': processor.orange_curve if hasattr(processor, 'orange_curve') else None,
           'normalized': processor.normalized_curve if hasattr(processor, 'normalized_curve') else None,
           'average': processor.average_curve if hasattr(processor, 'average_curve') else None,
           'hyperpol': processor.modified_hyperpol if hasattr(processor, 'modified_hyperpol') else None,
           'depol': processor.modified_depol if hasattr(processor, 'modified_depol') else None
       }

       for name, curve in curves.items():
           if curve is not None:
               min_val = np.min(curve)
               max_val = np.max(curve)
               std_val = np.std(curve)
               app_logger.info(f"{name} curve - min: {min_val:.2f}, max: {max_val:.2f}, std: {std_val:.2f}")
               
               if abs(min_val) > 5000 or abs(max_val) > 5000:
                   app_logger.warning(f"Extreme values in {name} curve")

   @staticmethod
   def validate_time_alignment(processor):
       """Verify time alignment between segments."""
       if not hasattr(processor, 'orange_curve_times'):
           return
       
       time_ranges = {}
       if hasattr(processor, 'modified_hyperpol_times'):
           time_ranges['hyperpol'] = (processor.modified_hyperpol_times[0], processor.modified_hyperpol_times[-1])
       if hasattr(processor, 'modified_depol_times'):
           time_ranges['depol'] = (processor.modified_depol_times[0], processor.modified_depol_times[-1])
       if hasattr(processor, 'average_curve_times'):
           time_ranges['average'] = (processor.average_curve_times[0], processor.average_curve_times[-1])

       for name, (start, end) in time_ranges.items():
           app_logger.info(f"{name} time range: {start*1000:.3f} - {end*1000:.3f} ms")

   @staticmethod
   def debug_processing_steps(processor):
       """Log detailed debugging info for each processing step."""
       steps = {
           'data_loaded': hasattr(processor, 'data') and processor.data is not None,
           'baseline_corrected': processor.processed_data is not None,
           'orange_generated': processor.orange_curve is not None,
           'normalized_created': hasattr(processor, 'normalized_curve'),
           'average_calculated': hasattr(processor, 'average_curve'),
           'peaks_modified': hasattr(processor, 'modified_hyperpol')
       }
       
       success = all(steps.values())
       app_logger.info("Processing Step Validation:")
       for step, completed in steps.items():
           app_logger.info(f"  {step}: {'✓' if completed else '✗'}")
       return success

   @staticmethod
   def validate_data_integrity(processor):
       """Validate data consistency across processing steps."""
       data_info = []
       
       if hasattr(processor, 'data'):
           data_info.append(f"Raw data: {len(processor.data)} points")
           data_info.append(f"  Range: [{np.min(processor.data):.2f}, {np.max(processor.data):.2f}] pA")
           data_info.append(f"  Sampling rate: {1.0/np.mean(np.diff(processor.time_data)):.2f} Hz")
       
       if hasattr(processor, 'processed_data'):
           data_info.append(f"Processed data: {len(processor.processed_data)} points")
           data_info.append(f"  Baseline: {processor.baseline:.2f} pA")
       
       app_logger.info("Data Integrity Check:")
       for info in data_info:
           app_logger.info(f"  {info}")

   @staticmethod
   def check_segment_transitions(processor):
       """Analyze transitions between segments for discontinuities."""
       bounds = processor.get_segment_bounds()
       
       for i, (start, end) in enumerate(bounds['normalized']):
           if i > 0:  # Check transition from previous segment
               prev_end = bounds['normalized'][i-1][1]
               if hasattr(processor, 'normalized_curve'):
                   jump = abs(processor.normalized_curve[start] - processor.normalized_curve[start-1])
                   app_logger.info(f"Segment {i} transition jump: {jump:.2f}")
                   if jump > 100:  # Arbitrary threshold
                       app_logger.warning(f"Large discontinuity at segment {i} transition")

   @staticmethod
   def analyze_signal_noise(processor):
       """Analyze noise characteristics in different signal components."""
       signals = {
           'raw': processor.data if hasattr(processor, 'data') else None,
           'processed': processor.processed_data if hasattr(processor, 'processed_data') else None,
           'orange': processor.orange_curve if hasattr(processor, 'orange_curve') else None
       }
       
       for name, signal in signals.items():
           if signal is not None:
               diff = np.diff(signal)
               noise_std = np.std(diff)
               snr = np.mean(np.abs(signal))/noise_std if noise_std > 0 else float('inf')
               
               app_logger.info(f"{name} signal noise analysis:")
               app_logger.info(f"  Noise std: {noise_std:.2f} pA")
               app_logger.info(f"  SNR: {snr:.2f}")
               
               z_scores = np.abs((diff - np.mean(diff))/np.std(diff))
               outliers = np.sum(z_scores > 3)
               if outliers > 0:
                   app_logger.warning(f"  Found {outliers} potential outliers in {name} signal")

   @staticmethod
   def verify_voltage_steps(processor):
       """Verify voltage step consistency across segments."""
       if not hasattr(processor, 'normalized_curve'):
           return
           
       bounds = processor.get_segment_bounds()
       v0 = processor.params.get('V0', -80)
       v1 = processor.params.get('V1', -100)
       v2 = processor.params.get('V2', -20)
       
       app_logger.info("\nVoltage Step Analysis:")
       app_logger.info(f"Configured steps: V0={v0}mV, V1={v1}mV, V2={v2}mV")
       
       for i, (start, end) in enumerate(bounds['normalized']):
           segment = processor.normalized_curve[start:end]
           mean_current = np.mean(segment)
           expected_voltage = v2 if i % 2 else v1
           voltage_diff = abs(expected_voltage - v0)
           
           conductance = mean_current/voltage_diff if voltage_diff != 0 else 0
           app_logger.info(f"Segment {i+1} conductance: {conductance:.2f} nS")

   @staticmethod
   def check_peaks(processor):
       """Validate peak detection and characteristics."""
       cycles = processor.cycles if hasattr(processor, 'cycles') else []
       if not cycles:
           app_logger.warning("No cycles detected")
           return
           
       peak_info = []
       for i, cycle in enumerate(cycles):
           peak_idx = np.argmin(cycle)  # Assuming negative peaks
           peak_val = cycle[peak_idx]
           baseline = np.median(cycle[:20])  # Use first 20 points as baseline
           amplitude = abs(peak_val - baseline)
           
           peak_info.append({
               'amplitude': amplitude,
               'baseline': baseline,
               'peak_value': peak_val
           })
           
           app_logger.info(f"\nCycle {i+1} characteristics:")
           app_logger.info(f"  Amplitude: {amplitude:.2f} pA")
           app_logger.info(f"  Baseline: {baseline:.2f} pA")

   @staticmethod
   def check_sampling_consistency(processor):
       """Verify sampling rate consistency across all time arrays."""
       time_arrays = {
           'main': processor.time_data,
           'orange': processor.orange_curve_times if hasattr(processor, 'orange_curve_times') else None,
           'normalized': processor.normalized_curve_times if hasattr(processor, 'normalized_curve_times') else None,
           'average': processor.average_curve_times if hasattr(processor, 'average_curve_times') else None
       }
       
       sampling_rates = {}
       for name, times in time_arrays.items():
           if times is not None:
               diff = np.diff(times)
               rate = 1.0/np.mean(diff)
               variance = np.std(diff)/np.mean(diff)
               sampling_rates[name] = {'rate': rate, 'variance': variance}
               
               if variance > 0.1:
                   app_logger.warning(f"Inconsistent sampling in {name}: {variance*100:.1f}% variation")
       return sampling_rates
   