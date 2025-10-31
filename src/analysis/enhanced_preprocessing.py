# src/analysis/enhanced_preprocessing.py
"""
Enhanced Preprocessing Module - Advanced signal preprocessing methods from ChaMa VB

This module implements sophisticated preprocessing techniques including:
- Multi-stage baseline correction
- Adaptive noise reduction
- Artifact detection and removal
- Signal quality assessment
- Advanced filtering methods
"""

import numpy as np
from scipy import signal, ndimage, interpolate
from scipy.optimize import minimize_scalar
from typing import Optional, Tuple, Dict, List, Union
import warnings
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import app_logger


class BaselineMethod(Enum):
    """Baseline correction methods"""
    MEAN = "mean"
    MEDIAN = "median"
    LINEAR_FIT = "linear_fit"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"
    PERCENTILE = "percentile"


class NoiseReductionMethod(Enum):
    """Noise reduction methods"""
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    SAVGOL = "savitzky_golay"
    WAVELET = "wavelet"
    ADAPTIVE = "adaptive"


@dataclass
class PreprocessingResults:
    """Results of preprocessing operations"""
    original_data: np.ndarray
    processed_data: np.ndarray
    baseline: Optional[np.ndarray] = None
    noise_estimate: Optional[float] = None
    quality_score: Optional[float] = None
    artifacts_detected: Optional[List[Tuple[int, int]]] = None
    processing_log: Optional[List[str]] = None


class SignalQualityAssessor:
    """Assess signal quality using multiple metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def assess_quality(self, data: np.ndarray, 
                      sampling_rate: float = 10000) -> Dict[str, float]:
        """
        Comprehensive signal quality assessment
        
        Args:
            data: Signal data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['snr'] = self._calculate_snr(data)
        metrics['noise_level'] = self._estimate_noise_level(data)
        metrics['dynamic_range'] = np.ptp(data)
        metrics['rms'] = np.sqrt(np.mean(data**2))
        
        # Frequency domain analysis
        freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(1024, len(data)//4))
        metrics['dominant_frequency'] = freqs[np.argmax(psd)]
        metrics['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        metrics['spectral_rolloff'] = self._spectral_rolloff(freqs, psd)
        
        # Artifact detection
        metrics['spike_count'] = self._count_spikes(data)
        metrics['drift_measure'] = self._measure_drift(data)
        metrics['discontinuity_count'] = self._count_discontinuities(data)
        
        # Overall quality score (0-1)
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            signal_power = np.var(data)
            # Estimate noise from high-frequency components
            b, a = signal.butter(4, 0.1, 'high')
            noise = signal.filtfilt(b, a, data)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            else:
                return 100.0  # Very high SNR
        except:
            return 0.0
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level using robust statistics"""
        # Use median absolute deviation for robust noise estimation
        median_data = np.median(data)
        mad = np.median(np.abs(data - median_data))
        return 1.4826 * mad  # Scale factor for Gaussian noise
    
    def _spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray, 
                         threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_idx = np.where(cumsum_psd >= threshold * total_energy)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _count_spikes(self, data: np.ndarray, threshold_factor: float = 5.0) -> int:
        """Count spike artifacts"""
        mad = np.median(np.abs(data - np.median(data)))
        threshold = threshold_factor * mad
        spikes = np.abs(data) > threshold
        return np.sum(spikes)
    
    def _measure_drift(self, data: np.ndarray) -> float:
        """Measure baseline drift"""
        # Linear detrend and measure residual
        x = np.arange(len(data))
        p = np.polyfit(x, data, 1)
        trend = np.polyval(p, x)
        return np.abs(p[0]) * len(data)  # Slope magnitude over entire trace
    
    def _count_discontinuities(self, data: np.ndarray, 
                              threshold_factor: float = 3.0) -> int:
        """Count discontinuities/jumps in the signal"""
        diff = np.diff(data)
        mad_diff = np.median(np.abs(diff - np.median(diff)))
        threshold = threshold_factor * mad_diff
        discontinuities = np.abs(diff) > threshold
        return np.sum(discontinuities)
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score (0-1)"""
        try:
            # Normalize and weight different metrics
            snr_score = min(1.0, max(0.0, (metrics['snr'] - 10) / 40))  # 10-50 dB range
            spike_score = max(0.0, 1.0 - metrics['spike_count'] / 100)
            drift_score = max(0.0, 1.0 - metrics['drift_measure'] / np.abs(metrics['rms']))
            
            # Weighted average
            weights = [0.4, 0.3, 0.3]  # SNR, spikes, drift
            scores = [snr_score, spike_score, drift_score]
            
            return np.average(scores, weights=weights)
        except:
            return 0.5  # Default moderate quality


class ArtifactDetector:
    """Detect and characterize various types of artifacts"""
    
    def __init__(self):
        self.detected_artifacts = []
        
    def detect_artifacts(self, data: np.ndarray, 
                        sampling_rate: float = 10000) -> List[Dict]:
        """
        Detect various types of artifacts in the signal
        
        Returns:
            List of detected artifacts with type, location, and confidence
        """
        artifacts = []
        
        # Spike artifacts
        artifacts.extend(self._detect_spikes(data, sampling_rate))
        
        # Baseline jumps
        artifacts.extend(self._detect_baseline_jumps(data))
        
        # Saturation artifacts
        artifacts.extend(self._detect_saturation(data))
        
        # Periodic noise
        artifacts.extend(self._detect_periodic_noise(data, sampling_rate))
        
        # Movement artifacts (low frequency)
        artifacts.extend(self._detect_movement_artifacts(data, sampling_rate))
        
        self.detected_artifacts = artifacts
        return artifacts
    
    def _detect_spikes(self, data: np.ndarray, sampling_rate: float) -> List[Dict]:
        """Detect spike artifacts"""
        artifacts = []
        
        # Use robust statistics for threshold
        mad = np.median(np.abs(data - np.median(data)))
        threshold = 5.0 * mad
        
        spike_indices = np.where(np.abs(data) > threshold)[0]
        
        # Group consecutive spikes
        if len(spike_indices) > 0:
            groups = []
            current_group = [spike_indices[0]]
            
            for i in range(1, len(spike_indices)):
                if spike_indices[i] - spike_indices[i-1] <= 3:  # Within 3 samples
                    current_group.append(spike_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [spike_indices[i]]
            groups.append(current_group)
            
            # Create artifact entries
            for group in groups:
                start_idx = group[0]
                end_idx = group[-1]
                magnitude = np.max(np.abs(data[start_idx:end_idx+1]))
                confidence = min(1.0, magnitude / threshold - 1.0)
                
                artifacts.append({
                    'type': 'spike',
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'magnitude': magnitude,
                    'confidence': confidence,
                    'duration_ms': (end_idx - start_idx) / sampling_rate * 1000
                })
        
        return artifacts
    
    def _detect_baseline_jumps(self, data: np.ndarray) -> List[Dict]:
        """Detect sudden baseline jumps"""
        artifacts = []
        
        # Look for sudden changes in local mean
        window_size = min(100, len(data) // 10)
        if window_size < 10:
            return artifacts
            
        local_means = ndimage.uniform_filter1d(data.astype(float), size=window_size)
        mean_diff = np.diff(local_means)
        
        # Robust threshold for jump detection
        mad_diff = np.median(np.abs(mean_diff - np.median(mean_diff)))
        threshold = 3.0 * mad_diff
        
        jump_indices = np.where(np.abs(mean_diff) > threshold)[0]
        
        for idx in jump_indices:
            magnitude = np.abs(mean_diff[idx])
            confidence = min(1.0, magnitude / threshold - 1.0)
            
            artifacts.append({
                'type': 'baseline_jump',
                'start_idx': idx,
                'end_idx': idx + 1,
                'magnitude': magnitude,
                'confidence': confidence
            })
        
        return artifacts
    
    def _detect_saturation(self, data: np.ndarray) -> List[Dict]:
        """Detect saturation artifacts"""
        artifacts = []
        
        # Look for regions where signal is at max/min for extended periods
        data_range = np.ptp(data)
        upper_threshold = np.max(data) - 0.01 * data_range
        lower_threshold = np.min(data) + 0.01 * data_range
        
        # Find consecutive points at limits
        upper_sat = data >= upper_threshold
        lower_sat = data <= lower_threshold
        
        for sat_mask, sat_type in [(upper_sat, 'upper'), (lower_sat, 'lower')]:
            # Find consecutive regions
            transitions = np.diff(sat_mask.astype(int))
            starts = np.where(transitions == 1)[0] + 1
            ends = np.where(transitions == -1)[0] + 1
            
            # Handle edge cases
            if sat_mask[0]:
                starts = np.concatenate([[0], starts])
            if sat_mask[-1]:
                ends = np.concatenate([ends, [len(data)]])
            
            for start, end in zip(starts, ends):
                if end - start > 5:  # At least 5 consecutive points
                    artifacts.append({
                        'type': f'saturation_{sat_type}',
                        'start_idx': start,
                        'end_idx': end,
                        'magnitude': data_range,
                        'confidence': 0.9
                    })
        
        return artifacts
    
    def _detect_periodic_noise(self, data: np.ndarray, 
                              sampling_rate: float) -> List[Dict]:
        """Detect periodic noise artifacts"""
        artifacts = []
        
        try:
            # FFT analysis to find dominant frequencies
            freqs, psd = signal.welch(data, fs=sampling_rate, 
                                    nperseg=min(1024, len(data)//4))
            
            # Look for prominent peaks that might be noise
            peaks, properties = signal.find_peaks(psd, height=np.mean(psd) * 3)
            
            for peak_idx in peaks:
                freq = freqs[peak_idx]
                power = psd[peak_idx]
                
                # Check if frequency is likely to be noise (50/60 Hz or harmonics)
                noise_freqs = [50, 60, 100, 120, 150, 180]
                is_noise = any(abs(freq - nf) < 2 for nf in noise_freqs)
                
                if is_noise and power > np.mean(psd) * 5:
                    artifacts.append({
                        'type': 'periodic_noise',
                        'frequency': freq,
                        'power': power,
                        'confidence': min(1.0, power / (np.mean(psd) * 10))
                    })
        except:
            pass  # Skip if FFT fails
            
        return artifacts
    
    def _detect_movement_artifacts(self, data: np.ndarray, 
                                  sampling_rate: float) -> List[Dict]:
        """Detect movement artifacts (low-frequency disturbances)"""
        artifacts = []
        
        try:
            # High-pass filter to isolate low-frequency components
            nyquist = sampling_rate / 2
            low_cutoff = 1.0  # 1 Hz
            
            if low_cutoff < nyquist:
                b, a = signal.butter(2, low_cutoff / nyquist, 'low')
                low_freq_component = signal.filtfilt(b, a, data)
                
                # Look for large low-frequency excursions
                mad = np.median(np.abs(low_freq_component - np.median(low_freq_component)))
                threshold = 3.0 * mad
                
                movement_mask = np.abs(low_freq_component) > threshold
                
                # Find consecutive regions
                transitions = np.diff(movement_mask.astype(int))
                starts = np.where(transitions == 1)[0] + 1
                ends = np.where(transitions == -1)[0] + 1
                
                # Handle edge cases
                if movement_mask[0]:
                    starts = np.concatenate([[0], starts])
                if movement_mask[-1]:
                    ends = np.concatenate([ends, [len(data)]])
                
                for start, end in zip(starts, ends):
                    duration_ms = (end - start) / sampling_rate * 1000
                    if duration_ms > 100:  # At least 100ms
                        magnitude = np.max(np.abs(low_freq_component[start:end]))
                        artifacts.append({
                            'type': 'movement_artifact',
                            'start_idx': start,
                            'end_idx': end,
                            'magnitude': magnitude,
                            'confidence': min(1.0, magnitude / threshold - 1.0),
                            'duration_ms': duration_ms
                        })
        except:
            pass  # Skip if filtering fails
            
        return artifacts


class EnhancedPreprocessor:
    """
    Enhanced preprocessing class with ChaMa VB-inspired methods
    """
    
    def __init__(self):
        self.quality_assessor = SignalQualityAssessor()
        self.artifact_detector = ArtifactDetector()
        self.processing_history = []
        
    def preprocess_signal(self, data: np.ndarray, 
                         time_data: Optional[np.ndarray] = None,
                         sampling_rate: float = 10000,
                         baseline_method: BaselineMethod = BaselineMethod.ADAPTIVE,
                         noise_reduction: Optional[NoiseReductionMethod] = None,
                         remove_artifacts: bool = True,
                         assess_quality: bool = True) -> PreprocessingResults:
        """
        Comprehensive signal preprocessing pipeline
        
        Args:
            data: Input signal data
            time_data: Optional time axis
            sampling_rate: Sampling rate in Hz
            baseline_method: Method for baseline correction
            noise_reduction: Optional noise reduction method
            remove_artifacts: Whether to detect and remove artifacts
            assess_quality: Whether to assess signal quality
            
        Returns:
            PreprocessingResults object with all results
        """
        original_data = data.copy()
        processed_data = data.copy()
        processing_log = []
        
        app_logger.info(f"Starting preprocessing of signal with {len(data)} points")
        
        # 1. Initial quality assessment
        quality_metrics = None
        if assess_quality:
            quality_metrics = self.quality_assessor.assess_quality(data, sampling_rate)
            processing_log.append(f"Initial quality score: {quality_metrics['overall_quality']:.3f}")
        
        # 2. Artifact detection
        artifacts = None
        if remove_artifacts:
            artifacts = self.artifact_detector.detect_artifacts(processed_data, sampling_rate)
            processing_log.append(f"Detected {len(artifacts)} artifacts")
        
        # 3. Baseline correction
        baseline = None
        processed_data, baseline = self._correct_baseline(
            processed_data, method=baseline_method
        )
        processing_log.append(f"Baseline corrected using {baseline_method.value} method")
        
        # 4. Artifact removal (if requested)
        if remove_artifacts and artifacts:
            processed_data = self._remove_artifacts(processed_data, artifacts)
            processing_log.append(f"Removed {len(artifacts)} artifacts")
        
        # 5. Noise reduction (if requested)
        noise_estimate = None
        if noise_reduction:
            processed_data, noise_estimate = self._reduce_noise(
                processed_data, method=noise_reduction, sampling_rate=sampling_rate
            )
            processing_log.append(f"Applied {noise_reduction.value} noise reduction")
        
        # 6. Final quality assessment
        final_quality = None
        if assess_quality:
            final_metrics = self.quality_assessor.assess_quality(processed_data, sampling_rate)
            final_quality = final_metrics['overall_quality']
            processing_log.append(f"Final quality score: {final_quality:.3f}")
        
        # Create results object
        results = PreprocessingResults(
            original_data=original_data,
            processed_data=processed_data,
            baseline=baseline,
            noise_estimate=noise_estimate,
            quality_score=final_quality,
            artifacts_detected=[(a['start_idx'], a['end_idx']) for a in artifacts] if artifacts else None,
            processing_log=processing_log
        )
        
        self.processing_history.append(results)
        app_logger.info(f"Preprocessing completed. Quality improved from "
                       f"{quality_metrics['overall_quality']:.3f} to {final_quality:.3f}" 
                       if assess_quality else "Preprocessing completed")
        
        return results
    
    def _correct_baseline(self, data: np.ndarray, 
                         method: BaselineMethod) -> Tuple[np.ndarray, np.ndarray]:
        """Apply baseline correction"""
        baseline = np.zeros_like(data)
        
        if method == BaselineMethod.MEAN:
            baseline[:] = np.mean(data)
            
        elif method == BaselineMethod.MEDIAN:
            baseline[:] = np.median(data)
            
        elif method == BaselineMethod.LINEAR_FIT:
            x = np.arange(len(data))
            p = np.polyfit(x, data, 1)
            baseline = np.polyval(p, x)
            
        elif method == BaselineMethod.POLYNOMIAL:
            x = np.arange(len(data))
            p = np.polyfit(x, data, 3)  # 3rd order polynomial
            baseline = np.polyval(p, x)
            
        elif method == BaselineMethod.ADAPTIVE:
            # Adaptive baseline using local percentiles
            window_size = min(1000, len(data) // 10)
            if window_size > 10:
                baseline = self._adaptive_baseline(data, window_size)
            else:
                baseline[:] = np.median(data)
                
        elif method == BaselineMethod.PERCENTILE:
            # Use 10th percentile as baseline
            baseline[:] = np.percentile(data, 10)
        
        corrected_data = data - baseline
        return corrected_data, baseline
    
    def _adaptive_baseline(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate adaptive baseline using sliding window"""
        baseline = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window_data = data[start:end]
            # Use 20th percentile of local window
            baseline[i] = np.percentile(window_data, 20)
        
        # Smooth the baseline
        baseline = ndimage.gaussian_filter1d(baseline, sigma=window_size/10)
        return baseline
    
    def _reduce_noise(self, data: np.ndarray, 
                     method: NoiseReductionMethod,
                     sampling_rate: float) -> Tuple[np.ndarray, float]:
        """Apply noise reduction"""
        noise_estimate = self.quality_assessor._estimate_noise_level(data)
        
        if method == NoiseReductionMethod.GAUSSIAN:
            # Gaussian smoothing
            sigma = max(1, sampling_rate / 10000)  # Adaptive sigma
            filtered_data = ndimage.gaussian_filter1d(data, sigma=sigma)
            
        elif method == NoiseReductionMethod.MEDIAN:
            # Median filtering
            kernel_size = max(3, int(sampling_rate / 5000))
            if kernel_size % 2 == 0:
                kernel_size += 1
            filtered_data = signal.medfilt(data, kernel_size=kernel_size)
            
        elif method == NoiseReductionMethod.SAVGOL:
            # Savitzky-Golay filter
            window_length = max(5, int(sampling_rate / 1000))
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, len(data) - 1)
            
            if window_length >= 5:
                filtered_data = signal.savgol_filter(data, window_length, 3)
            else:
                filtered_data = data  # Skip if window too small
                
        elif method == NoiseReductionMethod.BILATERAL:
            # Bilateral filter (edge-preserving)
            filtered_data = self._bilateral_filter(data, sigma_spatial=2, sigma_intensity=noise_estimate)
            
        elif method == NoiseReductionMethod.ADAPTIVE:
            # Adaptive filtering based on local noise characteristics
            filtered_data = self._adaptive_filter(data, sampling_rate)
            
        else:  # Default to Gaussian
            filtered_data = ndimage.gaussian_filter1d(data, sigma=1.0)
        
        return filtered_data, noise_estimate
    
    def _bilateral_filter(self, data: np.ndarray, 
                         sigma_spatial: float, 
                         sigma_intensity: float) -> np.ndarray:
        """Simple 1D bilateral filter for edge-preserving smoothing"""
        filtered_data = np.zeros_like(data)
        radius = int(3 * sigma_spatial)
        
        for i in range(len(data)):
            # Define spatial window
            start = max(0, i - radius)
            end = min(len(data), i + radius + 1)
            
            # Calculate spatial weights
            spatial_indices = np.arange(start, end)
            spatial_weights = np.exp(-0.5 * ((spatial_indices - i) / sigma_spatial) ** 2)
            
            # Calculate intensity weights
            intensity_diffs = np.abs(data[start:end] - data[i])
            intensity_weights = np.exp(-0.5 * (intensity_diffs / sigma_intensity) ** 2)
            
            # Combine weights
            weights = spatial_weights * intensity_weights
            weights /= np.sum(weights)
            
            # Apply filter
            filtered_data[i] = np.sum(weights * data[start:end])
        
        return filtered_data
    
    def _adaptive_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Adaptive filtering based on local signal characteristics"""
        # Estimate local noise level using sliding window
        window_size = max(100, int(sampling_rate / 100))  # 10ms windows
        filtered_data = np.zeros_like(data)
        
        for i in range(0, len(data), window_size // 2):
            start = i
            end = min(len(data), i + window_size)
            window_data = data[start:end]
            
            # Estimate local noise
            local_noise = self.quality_assessor._estimate_noise_level(window_data)
            
            # Adaptive smoothing based on noise level
            if local_noise > np.std(data) * 0.1:  # High noise region
                sigma = 2.0
            else:  # Low noise region
                sigma = 0.5
            
            # Apply Gaussian smoothing
            smoothed_window = ndimage.gaussian_filter1d(window_data, sigma=sigma)
            filtered_data[start:end] = smoothed_window
        
        return filtered_data
    
    def _remove_artifacts(self, data: np.ndarray, 
                         artifacts: List[Dict]) -> np.ndarray:
        """Remove detected artifacts using interpolation"""
        cleaned_data = data.copy()
        
        for artifact in artifacts:
            if 'start_idx' in artifact and 'end_idx' in artifact:
                start_idx = artifact['start_idx']
                end_idx = artifact['end_idx']
                
                # Interpolate over artifact region
                if start_idx > 0 and end_idx < len(data) - 1:
                    # Use surrounding points for interpolation
                    x_points = [start_idx - 1, end_idx + 1]
                    y_points = [data[start_idx - 1], data[end_idx + 1]]
                    
                    x_interp = np.arange(start_idx, end_idx + 1)
                    y_interp = np.interp(x_interp, x_points, y_points)
                    
                    cleaned_data[start_idx:end_idx + 1] = y_interp
        
        return cleaned_data
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of all preprocessing operations"""
        if not self.processing_history:
            return {"message": "No preprocessing performed yet"}
        
        latest = self.processing_history[-1]
        return {
            "total_operations": len(self.processing_history),
            "latest_quality_score": latest.quality_score,
            "artifacts_detected": len(latest.artifacts_detected) if latest.artifacts_detected else 0,
            "noise_estimate": latest.noise_estimate,
            "processing_steps": latest.processing_log
        }