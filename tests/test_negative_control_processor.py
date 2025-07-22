# tests/test_negative_control_processor.py
"""
Comprehensive test suite for the Enhanced Negative Control Processor
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add src to path for testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.negative_control_processor import NegativeControlProcessor, ProtocolParameters


class TestNegativeControlProcessor:
    """Test suite for NegativeControlProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = NegativeControlProcessor()
        self.sample_rate = 10000  # 10 kHz
        self.time_step = 1.0 / self.sample_rate * 1000  # in ms
        
        # Create synthetic test data
        self.duration = 100  # ms
        self.num_points = int(self.duration / self.time_step)
        self.time_data = np.linspace(0, self.duration, self.num_points)
        
        # Synthetic current trace with baseline, control pulses, and test pulse
        self.current_trace = self._create_synthetic_current()
        
        # Setup realistic protocol parameters
        self.test_params = {
            'baseline_duration': 10.0,  # ms
            'neg_control_on_duration': 20.0,  # ms
            'neg_control_off_duration': 20.0,  # ms
            'num_control_traces': 2,
            'test_pulse_v1': -80.0,  # mV
            'test_pulse_v2': -40.0,  # mV
            'test_pulse_v3': -80.0,  # mV
            'test_on_duration': 15.0,  # ms
            'test_off_duration': 15.0,  # ms
            'sampling_interval': self.time_step,
            'neg_control_v1': -80.0,  # mV
            'neg_control_v2': -40.0   # mV
        }
        
    def _create_synthetic_current(self):
        """Create synthetic current trace for testing"""
        current = np.zeros(self.num_points)
        
        # Add baseline offset
        baseline = -0.1  # nA
        current += baseline
        
        # Add noise
        noise = np.random.normal(0, 0.01, self.num_points)
        current += noise
        
        # Add control pulses (simplified square pulses)
        control_start = int(10 / self.time_step)  # After 10ms baseline
        control_duration = int(20 / self.time_step)  # 20ms ON
        
        # First control pulse
        current[control_start:control_start + control_duration] += 0.5
        # OFF phase
        current[control_start + control_duration:control_start + 2*control_duration] -= 0.3
        
        # Second control pulse
        second_start = control_start + 2*control_duration
        current[second_start:second_start + control_duration] += 0.5
        current[second_start + control_duration:second_start + 2*control_duration] -= 0.3
        
        # Add test pulse
        test_start = second_start + 2*control_duration
        test_duration = int(15 / self.time_step)
        if test_start + 2*test_duration < len(current):
            current[test_start:test_start + test_duration] += 1.0  # ON
            current[test_start + test_duration:test_start + 2*test_duration] -= 0.8  # OFF
        
        return current
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor.negative_control is None
        assert self.processor.stored_negative_control is None
        assert self.processor.original_current is None
        assert isinstance(self.processor.protocol_params, ProtocolParameters)
        
    def test_set_protocol_parameters(self):
        """Test setting protocol parameters"""
        self.processor.set_protocol_parameters(self.test_params)
        
        assert self.processor.protocol_params.baseline_duration == 10.0
        assert self.processor.protocol_params.num_control_traces == 2
        assert self.processor.protocol_params.test_pulse_v1 == -80.0
        
    def test_average_control_pulses_basic(self):
        """Test basic negative control averaging"""
        self.processor.set_protocol_parameters(self.test_params)
        
        result = self.processor.average_control_pulses(self.current_trace, self.time_data)
        
        assert result is not None
        assert len(result) > 0
        assert self.processor.negative_control is not None
        assert self.processor.original_current is not None
        assert np.array_equal(self.processor.original_current, self.current_trace)
        
    def test_average_control_pulses_baseline_subtraction(self):
        """Test that baseline is properly subtracted"""
        self.processor.set_protocol_parameters(self.test_params)
        
        # Create trace with known baseline
        trace_with_baseline = self.current_trace + 2.0  # Add 2 nA baseline
        
        result = self.processor.average_control_pulses(trace_with_baseline)
        
        # Check that baseline was subtracted
        baseline_points = int(10.0 / self.time_step)  # 10ms baseline
        original_baseline = np.mean(trace_with_baseline[:baseline_points])
        processed_baseline = np.mean(self.processor.processed_current[:baseline_points])
        
        assert abs(processed_baseline) < abs(original_baseline)
        
    def test_average_control_pulses_empty_input(self):
        """Test error handling for empty input"""
        with pytest.raises(ValueError, match="Current trace cannot be empty"):
            self.processor.average_control_pulses(np.array([]))
            
    def test_remove_ohmic_component(self):
        """Test ohmic component removal"""
        # Create trace with linear trend
        linear_trace = np.linspace(0, 1, 1000) + np.random.normal(0, 0.01, 1000)
        
        result = self.processor.remove_ohmic_component(linear_trace)
        
        assert result is not None
        assert len(result) == len(linear_trace)
        assert self.processor.ohmic_fit_params is not None
        assert len(self.processor.ohmic_fit_params) == 2  # slope and intercept
        
        # Check that linear trend was removed
        assert np.std(result) < np.std(linear_trace)
        
    def test_remove_ohmic_component_custom_range(self):
        """Test ohmic component removal with custom fit range"""
        linear_trace = np.linspace(0, 1, 1000)
        fit_range = (100, 200)
        
        result = self.processor.remove_ohmic_component(linear_trace, fit_range)
        
        assert result is not None
        assert self.processor.ohmic_fit_params is not None
        
    def test_calculate_charge_movement(self):
        """Test charge movement calculation"""
        self.processor.set_protocol_parameters(self.test_params)
        
        # First calculate negative control
        self.processor.average_control_pulses(self.current_trace)
        
        # Then calculate charge movement
        result = self.processor.calculate_charge_movement(self.current_trace, mode="trace")
        
        assert result is not None
        assert self.processor.charge_movement is not None
        assert len(result) == len(self.current_trace)
        
    def test_calculate_charge_movement_different_modes(self):
        """Test charge movement calculation with different modes"""
        self.processor.set_protocol_parameters(self.test_params)
        self.processor.average_control_pulses(self.current_trace)
        
        modes = ["trace", "on", "off", "average"]
        
        for mode in modes:
            result = self.processor.calculate_charge_movement(self.current_trace, mode=mode)
            assert result is not None
            assert len(result) == len(self.current_trace)
            
    def test_calculate_charge_movement_no_negative_control(self):
        """Test error when calculating charge movement without negative control"""
        with pytest.raises(ValueError, match="Negative control must be calculated first"):
            self.processor.calculate_charge_movement(self.current_trace)
            
    def test_add_stored_negative_control(self):
        """Test adding stored negative control"""
        self.processor.set_protocol_parameters(self.test_params)
        self.processor.average_control_pulses(self.current_trace)
        
        # Create synthetic stored control
        stored_control = np.random.normal(0, 0.1, len(self.processor.negative_control))
        
        result = self.processor.add_stored_negative_control(stored_control, weight=0.5)
        
        assert result is not None
        assert self.processor.stored_negative_control is not None
        assert np.array_equal(self.processor.stored_negative_control, stored_control)
        
    def test_add_stored_negative_control_no_current(self):
        """Test adding stored control when no current control exists"""
        stored_control = np.random.normal(0, 0.1, 100)
        
        result = self.processor.add_stored_negative_control(stored_control)
        
        assert np.array_equal(result, stored_control)
        
    def test_get_processing_summary(self):
        """Test processing summary generation"""
        self.processor.set_protocol_parameters(self.test_params)
        
        # Initial summary
        summary = self.processor.get_processing_summary()
        assert not summary["has_negative_control"]
        assert not summary["has_charge_movement"]
        
        # After processing
        self.processor.average_control_pulses(self.current_trace)
        self.processor.calculate_charge_movement(self.current_trace)
        
        summary = self.processor.get_processing_summary()
        assert summary["has_negative_control"]
        assert summary["has_charge_movement"]
        assert "negative_control_stats" in summary
        assert "protocol_parameters" in summary
        
    def test_protocol_parameters_dataclass(self):
        """Test ProtocolParameters dataclass"""
        params = ProtocolParameters()
        assert params.baseline_duration == 0.0
        assert params.num_control_traces == 1
        
        # Test setting values
        params.baseline_duration = 15.0
        assert params.baseline_duration == 15.0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.processor = NegativeControlProcessor()
        
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline"""
        # Setup realistic parameters
        params = {
            'baseline_duration': 10.0,
            'neg_control_on_duration': 25.0,
            'neg_control_off_duration': 25.0,
            'num_control_traces': 3,
            'test_pulse_v1': -80.0,
            'test_pulse_v2': 0.0,
            'test_pulse_v3': -80.0,
            'test_on_duration': 20.0,
            'test_off_duration': 20.0,
            'sampling_interval': 0.1,
            'neg_control_v1': -80.0,
            'neg_control_v2': 0.0
        }
        
        # Create realistic current trace
        duration_ms = 200
        points = int(duration_ms / params['sampling_interval'])
        current = np.zeros(points)
        
        # Add baseline
        current += -0.05
        
        # Add control and test pulses (simplified)
        current[100:600] += 0.3  # Control pulse 1
        current[700:1200] += 0.3  # Control pulse 2
        current[1300:1800] += 0.3  # Control pulse 3
        current[1900:2100] += 1.0  # Test pulse
        
        # Add noise
        current += np.random.normal(0, 0.02, points)
        
        # Run full pipeline
        self.processor.set_protocol_parameters(params)
        
        # Step 1: Average control pulses
        neg_control = self.processor.average_control_pulses(current)
        assert neg_control is not None
        
        # Step 2: Remove ohmic component from negative control
        corrected_neg_control = self.processor.remove_ohmic_component(neg_control)
        assert corrected_neg_control is not None
        
        # Step 3: Calculate charge movement
        charge_movement = self.processor.calculate_charge_movement(current, mode="average")
        assert charge_movement is not None
        
        # Step 4: Get summary
        summary = self.processor.get_processing_summary()
        assert summary["has_negative_control"]
        assert summary["has_charge_movement"]
        assert summary["has_ohmic_params"]
        
    def test_error_recovery(self):
        """Test error recovery and graceful degradation"""
        # Test with minimal data
        minimal_current = np.array([1, 2, 3, 4, 5])
        
        self.processor.set_protocol_parameters({
            'baseline_duration': 1.0,
            'sampling_interval': 1.0,
            'num_control_traces': 1
        })
        
        # Should handle gracefully
        result = self.processor.average_control_pulses(minimal_current)
        assert result is not None


# Performance tests
class TestPerformance:
    """Performance tests for large datasets"""
    
    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_processing_speed(self, data_size):
        """Test processing speed with different data sizes"""
        import time
        
        processor = NegativeControlProcessor()
        current = np.random.normal(0, 1, data_size)
        
        params = {
            'baseline_duration': 10.0,
            'sampling_interval': 0.1,
            'num_control_traces': 2,
            'neg_control_on_duration': 20.0,
            'neg_control_off_duration': 20.0
        }
        processor.set_protocol_parameters(params)
        
        start_time = time.time()
        result = processor.average_control_pulses(current)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])