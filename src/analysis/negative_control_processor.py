# src/analysis/negative_control_processor.py
"""
Enhanced Negative Control Processor - Ported from ChaMa VB version

This module handles negative control traces, averaging, baseline subtraction,
and ohmic component removal as implemented in the original ChaMa VB application.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from src.utils.logger import app_logger

@dataclass
class ProtocolParameters:
    """Protocol parameters equivalent to p_data array in ChaMa VB"""
    baseline_duration: float = 0.0      # p_data[2] - baseline duration in ms
    neg_control_on_duration: float = 0.0   # p_data[4] - negative control ON duration
    neg_control_off_duration: float = 0.0  # p_data[5] - negative control OFF duration
    num_control_traces: int = 1         # p_data[6] - number of control traces
    test_pulse_v1: float = 0.0          # p_data[7] - test pulse voltage 1
    test_pulse_v2: float = 0.0          # p_data[9] - test pulse voltage 2
    test_pulse_v3: float = 0.0          # p_data[11] - test pulse voltage 3
    test_on_duration: float = 0.0       # p_data[10] - test pulse ON duration
    test_off_duration: float = 0.0      # p_data[13] - test pulse OFF duration
    sampling_interval: float = 0.1      # p_data[22] - sampling interval in ms
    neg_control_v1: float = 0.0         # p_data[1] - negative control voltage 1
    neg_control_v2: float = 0.0         # p_data[3] - negative control voltage 2

class NegativeControlProcessor:
    """
    Negative Control Processor with methods ported from ChaMa VB version.
    
    Handles:
    - Baseline subtraction
    - Negative control averaging
    - Ohmic component removal
    - Charge movement calculation
    """
    
    def __init__(self):
        self.negative_control: Optional[np.ndarray] = None
        self.stored_negative_control: Optional[np.ndarray] = None
        self.original_current: Optional[np.ndarray] = None
        self.processed_current: Optional[np.ndarray] = None
        self.charge_movement: Optional[np.ndarray] = None
        self.ohmic_fit_params: Optional[np.ndarray] = None
        self.protocol_params = ProtocolParameters()
        
    def set_protocol_parameters(self, params: Dict[str, float]):
        """Set protocol parameters from dictionary"""
        for key, value in params.items():
            if hasattr(self.protocol_params, key):
                setattr(self.protocol_params, key, value)
        app_logger.info(f"Protocol parameters updated: {params}")
    
    def average_control_pulses(self, current_trace: np.ndarray, 
                             time_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Average negative control pulses and subtract baseline.
        Ported from ChaMa VB AverageControlPulsesToolStripMenuItem_Click
        
        Args:
            current_trace: Raw current data (equivalent to Cur[] in VB)
            time_data: Optional time data for validation
            
        Returns:
            Averaged negative control trace
        """
        if current_trace is None or len(current_trace) == 0:
            raise ValueError("Current trace cannot be empty")
            
        # Store original current
        self.original_current = current_trace.copy()
        
        # Calculate baseline length in points
        baseline_points = int(self.protocol_params.baseline_duration / 
                            self.protocol_params.sampling_interval)
        
        if baseline_points >= len(current_trace):
            baseline_points = len(current_trace) // 4  # Fallback
            
        # Calculate and subtract baseline
        baseline = np.mean(current_trace[:baseline_points])
        corrected_trace = current_trace - baseline
        
        app_logger.info(f"Baseline calculated: {baseline:.4f} nA from {baseline_points} points")
        
        # Calculate negative control pulse length
        pulse_duration = (self.protocol_params.neg_control_on_duration + 
                         self.protocol_params.neg_control_off_duration)
        pulse_points = int(pulse_duration / self.protocol_params.sampling_interval)
        
        if pulse_points <= 0:
            pulse_points = len(corrected_trace) // max(1, self.protocol_params.num_control_traces)
            
        # Average control traces
        num_traces = self.protocol_params.num_control_traces
        averaged_control = np.zeros(pulse_points)
        
        start_idx = baseline_points
        for trace_idx in range(num_traces):
            end_idx = start_idx + pulse_points
            if end_idx <= len(corrected_trace):
                averaged_control += corrected_trace[start_idx:end_idx]
                start_idx = end_idx
            else:
                app_logger.warning(f"Trace {trace_idx} extends beyond data length")
                break
                
        if num_traces > 0:
            averaged_control /= num_traces
            
        # Store result
        self.negative_control = averaged_control
        self.processed_current = corrected_trace
        
        app_logger.info(f"Averaged {num_traces} control traces, "
                       f"resulting in {len(averaged_control)} points")
        
        return averaged_control
    
    def remove_ohmic_component(self, trace: np.ndarray, 
                              fit_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Remove ohmic component using linear fit.
        Ported from ChaMa VB RemoveOhmicComponentToolStripMenuItem_Click
        
        Args:
            trace: Input trace for ohmic component removal
            fit_range: Optional range for linear fit (start_idx, end_idx)
            
        Returns:
            Trace with ohmic component removed
        """
        if trace is None or len(trace) == 0:
            raise ValueError("Trace cannot be empty")
            
        # Determine fit range
        if fit_range is None:
            # Use first and last 10% of trace for linear fit
            range_size = len(trace) // 10
            fit_range = (0, range_size, len(trace) - range_size, len(trace))
        else:
            fit_range = (fit_range[0], fit_range[1], fit_range[0], fit_range[1])
            
        # Extract data for linear fit
        x_fit = np.concatenate([
            np.arange(fit_range[0], fit_range[1]),
            np.arange(fit_range[2], fit_range[3])
        ])
        y_fit = np.concatenate([
            trace[fit_range[0]:fit_range[1]],
            trace[fit_range[2]:fit_range[3]]
        ])
        
        if len(x_fit) < 2:
            app_logger.warning("Insufficient points for linear fit")
            return trace
            
        # Perform linear fit
        fit_params = np.polyfit(x_fit, y_fit, 1)
        self.ohmic_fit_params = fit_params
        
        # Generate ohmic component
        x_full = np.arange(len(trace))
        ohmic_component = np.polyval(fit_params, x_full)
        
        # Subtract ohmic component
        corrected_trace = trace - ohmic_component
        
        app_logger.info(f"Ohmic component removed: slope={fit_params[0]:.6f}, "
                       f"intercept={fit_params[1]:.6f}")
        
        return corrected_trace
    
    def calculate_charge_movement(self, current_trace: np.ndarray,
                                mode: str = "trace") -> np.ndarray:
        """
        Calculate charge movement with linear capacitive correction.
        Ported from ChaMa VB LinearCapacitiveCorrectionToolStripMenuItem_Click
        
        Args:
            current_trace: Current trace data
            mode: Correction mode ("trace", "on", "off", "average")
            
        Returns:
            Charge movement trace
        """
        if self.negative_control is None:
            raise ValueError("Negative control must be calculated first")
            
        if current_trace is None or len(current_trace) == 0:
            raise ValueError("Current trace cannot be empty")
            
        # Calculate pulse durations in points
        on_duration = self.protocol_params.test_on_duration
        off_duration = self.protocol_params.test_off_duration
        
        on_points = int(on_duration / self.protocol_params.sampling_interval)
        off_points = int(off_duration / self.protocol_params.sampling_interval)
        
        # Calculate voltage steps
        dv_on = self.protocol_params.test_pulse_v2 - self.protocol_params.test_pulse_v1
        dv_off = self.protocol_params.test_pulse_v3 - self.protocol_params.test_pulse_v2
        
        # Generate control arrays
        max_points = max(on_points, off_points)
        control_duration = int((self.protocol_params.neg_control_on_duration) / 
                              self.protocol_params.sampling_interval)
        
        # Scale negative control for voltage steps
        c_on = np.zeros(max_points)
        c_off = np.zeros(max_points)
        
        if control_duration > 0 and len(self.negative_control) >= control_duration:
            scaling_on = dv_on / (self.protocol_params.neg_control_v2 - 
                                 self.protocol_params.neg_control_v1)
            scaling_off = dv_off / (self.protocol_params.neg_control_v2 - 
                                   self.protocol_params.neg_control_v1)
                                   
            c_on[:min(control_duration, max_points)] = (
                self.negative_control[:min(control_duration, max_points)] * scaling_on
            )
            c_off[:min(control_duration, max_points)] = (
                self.negative_control[control_duration:2*control_duration][:min(control_duration, max_points)] * scaling_off
            )
        
        # Initialize charge movement
        charge_movement = current_trace.copy()
        total_points = on_points + off_points
        
        if len(charge_movement) < total_points:
            app_logger.warning("Current trace shorter than expected pulse duration")
            total_points = len(charge_movement)
            on_points = min(on_points, total_points // 2)
            off_points = total_points - on_points
        
        # Apply correction based on mode
        if mode == "trace":
            # Subtract ON control for ON phase, add OFF control for OFF phase
            charge_movement[:on_points] -= c_on[:on_points]
            if on_points + off_points <= len(charge_movement):
                charge_movement[on_points:on_points + off_points] += c_off[:off_points]
                
        elif mode == "on":
            # Use ON control for both phases
            charge_movement[:on_points] -= c_on[:on_points]
            if on_points + off_points <= len(charge_movement):
                charge_movement[on_points:on_points + off_points] += c_on[:off_points]
                
        elif mode == "off":
            # Use OFF control for both phases
            charge_movement[:on_points] -= c_off[:on_points]
            if on_points + off_points <= len(charge_movement):
                charge_movement[on_points:on_points + off_points] += c_off[:off_points]
                
        elif mode == "average":
            # Use average of ON and OFF controls
            avg_control = (c_on + c_off) / 2
            charge_movement[:on_points] -= avg_control[:on_points]
            if on_points + off_points <= len(charge_movement):
                charge_movement[on_points:on_points + off_points] += avg_control[:off_points]
        
        # Store result
        self.charge_movement = charge_movement
        
        app_logger.info(f"Charge movement calculated using {mode} mode")
        
        return charge_movement
    
    def add_stored_negative_control(self, stored_control: np.ndarray,
                                  weight: float = 1.0) -> np.ndarray:
        """
        Add stored negative control to current negative control.
        Ported from ChaMa VB AddImportedToolStripMenuItem_Click
        
        Args:
            stored_control: Previously stored negative control
            weight: Weight for adding stored control
            
        Returns:
            Combined negative control
        """
        if self.negative_control is None:
            app_logger.warning("No current negative control available")
            return stored_control.copy()
            
        # Ensure both arrays have same length
        min_length = min(len(self.negative_control), len(stored_control))
        
        combined_control = np.zeros(min_length)
        combined_control[:min_length] = (
            self.negative_control[:min_length] + 
            weight * stored_control[:min_length]
        )
        
        self.stored_negative_control = stored_control.copy()
        self.negative_control = combined_control
        
        app_logger.info(f"Added stored negative control with weight {weight}")
        
        return combined_control
    
    def get_processing_summary(self) -> Dict[str, any]:
        """Get summary of processing steps performed"""
        summary = {
            "has_negative_control": self.negative_control is not None,
            "has_stored_control": self.stored_negative_control is not None,
            "has_charge_movement": self.charge_movement is not None,
            "has_ohmic_params": self.ohmic_fit_params is not None,
            "protocol_parameters": {
                "baseline_duration": self.protocol_params.baseline_duration,
                "num_control_traces": self.protocol_params.num_control_traces,
                "sampling_interval": self.protocol_params.sampling_interval
            }
        }
        
        if self.negative_control is not None:
            summary["negative_control_length"] = len(self.negative_control)
            summary["negative_control_stats"] = {
                "mean": float(np.mean(self.negative_control)),
                "std": float(np.std(self.negative_control)),
                "min": float(np.min(self.negative_control)),
                "max": float(np.max(self.negative_control))
            }
            
        if self.ohmic_fit_params is not None:
            summary["ohmic_fit"] = {
                "slope": float(self.ohmic_fit_params[0]),
                "intercept": float(self.ohmic_fit_params[1])
            }
            
        return summary