# src/analysis/enhanced_ai_features.py
"""
Enhanced Feature Engineering for AI Curve Learning

This module extends the basic feature extraction with advanced signal processing
techniques to capture more nuanced curve characteristics that experts implicitly use.
"""

import numpy as np
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import pywt
from src.utils.logger import app_logger

class EnhancedFeatureExtractor:
    """Advanced feature extraction for curve analysis AI."""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, time, current):
        """
        Extract comprehensive feature set from curve data.
        
        Returns dict with all extracted features.
        """
        features = {}
        
        # Basic features (keep existing)
        features.update(self._extract_basic_features(time, current))
        
        # Advanced features
        features.update(self._extract_wavelet_features(current))
        features.update(self._extract_frequency_features(time, current))
        features.update(self._extract_shape_complexity_features(time, current))
        features.update(self._extract_critical_points_features(time, current))
        features.update(self._extract_phase_space_features(time, current))
        
        self.feature_names = list(features.keys())
        return features
    
    def _extract_basic_features(self, time, current):
        """Extract basic statistical features."""
        return {
            'mean': np.mean(current),
            'std': np.std(current),
            'median': np.median(current),
            'mad': np.median(np.abs(current - np.median(current))),
            'iqr': np.percentile(current, 75) - np.percentile(current, 25),
            'cv': np.std(current) / np.mean(current) if np.mean(current) != 0 else 0,
            'range': np.ptp(current),
            'energy': np.sum(current ** 2),
            'rms': np.sqrt(np.mean(current ** 2))
        }
    
    def _extract_wavelet_features(self, current):
        """Extract wavelet-based features for multi-scale analysis."""
        features = {}
        
        try:
            # Perform wavelet decomposition
            wavelet = 'db4'
            max_level = min(5, pywt.dwt_max_level(len(current), wavelet))
            coeffs = pywt.wavedec(current, wavelet, level=max_level)
            
            # Energy distribution across scales
            total_energy = sum(np.sum(c**2) for c in coeffs)
            
            for i, c in enumerate(coeffs):
                level_name = 'approx' if i == 0 else f'detail_{i}'
                energy = np.sum(c**2)
                features[f'wavelet_{level_name}_energy'] = energy
                features[f'wavelet_{level_name}_energy_ratio'] = energy / total_energy if total_energy > 0 else 0
                features[f'wavelet_{level_name}_std'] = np.std(c)
            
            # Wavelet entropy
            energies = [np.sum(c**2) for c in coeffs]
            features['wavelet_entropy'] = entropy(energies) if sum(energies) > 0 else 0
            
        except Exception as e:
            app_logger.warning(f"Wavelet feature extraction failed: {str(e)}")
            features.update({f'wavelet_{k}': 0 for k in ['approx_energy', 'detail_1_energy', 'entropy']})
        
        return features
    
    def _extract_frequency_features(self, time, current):
        """Extract frequency domain features."""
        features = {}
        
        try:
            # Compute FFT
            n = len(current)
            dt = np.mean(np.diff(time))
            yf = fft(current)
            xf = fftfreq(n, dt)[:n//2]
            power = 2.0/n * np.abs(yf[:n//2])
            
            # Dominant frequency
            dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
            features['dominant_frequency'] = xf[dominant_idx]
            features['dominant_power'] = power[dominant_idx]
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(xf * power) / np.sum(power) if np.sum(power) > 0 else 0
            
            # Spectral bandwidth
            sc = features['spectral_centroid']
            features['spectral_bandwidth'] = np.sqrt(np.sum(((xf - sc) ** 2) * power) / np.sum(power)) if np.sum(power) > 0 else 0
            
            # Power in different frequency bands
            total_power = np.sum(power)
            if total_power > 0:
                # Define frequency bands (adjust based on your sampling rate)
                bands = [(0, 10), (10, 50), (50, 100), (100, 500)]
                for low, high in bands:
                    band_mask = (xf >= low) & (xf < high)
                    band_power = np.sum(power[band_mask])
                    features[f'power_{low}_{high}Hz'] = band_power / total_power
            
        except Exception as e:
            app_logger.warning(f"Frequency feature extraction failed: {str(e)}")
            features.update({'dominant_frequency': 0, 'spectral_centroid': 0})
        
        return features
    
    def _extract_shape_complexity_features(self, time, current):
        """Extract features related to curve shape complexity."""
        features = {}
        
        try:
            # Sample entropy (measure of signal complexity)
            features['sample_entropy'] = self._sample_entropy(current, 2, 0.2 * np.std(current))
            
            # Zero crossing rate
            mean_centered = current - np.mean(current)
            zero_crossings = np.where(np.diff(np.sign(mean_centered)))[0]
            features['zero_crossing_rate'] = len(zero_crossings) / len(current)
            
            # Curve length (arc length)
            dx = np.diff(time)
            dy = np.diff(current)
            arc_length = np.sum(np.sqrt(dx**2 + dy**2))
            features['normalized_arc_length'] = arc_length / (time[-1] - time[0])
            
            # Fractal dimension (using box counting approximation)
            features['approx_fractal_dim'] = self._approximate_fractal_dimension(time, current)
            
        except Exception as e:
            app_logger.warning(f"Shape complexity feature extraction failed: {str(e)}")
            
        return features
    
    def _extract_critical_points_features(self, time, current):
        """Extract features related to critical points and curve dynamics."""
        features = {}
        
        try:
            # First derivative
            dt = np.diff(time)
            dy = np.diff(current)
            first_deriv = dy / dt
            
            # Second derivative
            d2y = np.diff(dy)
            dt2 = dt[:-1]
            second_deriv = d2y / dt2
            
            # Find critical points (where first derivative ≈ 0)
            critical_threshold = 0.01 * np.std(first_deriv)
            critical_points = np.where(np.abs(first_deriv) < critical_threshold)[0]
            features['num_critical_points'] = len(critical_points)
            
            # Find inflection points (where second derivative changes sign)
            inflection_points = np.where(np.diff(np.sign(second_deriv)))[0]
            features['num_inflection_points'] = len(inflection_points)
            
            # Curvature statistics
            curvature = np.abs(second_deriv) / (1 + first_deriv[:-1]**2)**(3/2)
            features['max_curvature'] = np.max(curvature) if len(curvature) > 0 else 0
            features['mean_curvature'] = np.mean(curvature) if len(curvature) > 0 else 0
            
            # Phase space features
            if len(critical_points) > 0:
                # Time between critical points
                if len(critical_points) > 1:
                    critical_intervals = np.diff(time[critical_points])
                    features['mean_critical_interval'] = np.mean(critical_intervals)
                    features['std_critical_interval'] = np.std(critical_intervals)
                else:
                    features['mean_critical_interval'] = 0
                    features['std_critical_interval'] = 0
            
        except Exception as e:
            app_logger.warning(f"Critical points feature extraction failed: {str(e)}")
            
        return features
    
    def _extract_phase_space_features(self, time, current):
        """Extract phase space reconstruction features."""
        features = {}
        
        try:
            # Create phase space embedding
            tau = self._estimate_time_delay(current)
            embedding_dim = 3
            
            if tau > 0 and len(current) > tau * embedding_dim:
                # Reconstruct phase space
                phase_space = self._phase_space_reconstruction(current, tau, embedding_dim)
                
                # Calculate phase space properties
                # Centroid
                centroid = np.mean(phase_space, axis=0)
                features['phase_space_spread'] = np.mean(np.linalg.norm(phase_space - centroid, axis=1))
                
                # Phase space volume (using convex hull would be better but computationally expensive)
                features['phase_space_volume'] = np.prod(np.ptp(phase_space, axis=0))
                
        except Exception as e:
            app_logger.warning(f"Phase space feature extraction failed: {str(e)}")
            
        return features
    
    def _sample_entropy(self, data, m, r):
        """Calculate sample entropy of the signal."""
        N = len(data)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            C = 0
            for i in range(len(patterns)):
                matching = 0
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        matching += 1
                if matching > 0:
                    C += np.log(matching / (N - m))
            return C / (N - m + 1)
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0
    
    def _approximate_fractal_dimension(self, time, current):
        """Approximate fractal dimension using variation method."""
        try:
            # Normalize data
            x = (time - time[0]) / (time[-1] - time[0])
            y = (current - np.min(current)) / (np.max(current) - np.min(current))
            
            # Calculate variations at different scales
            scales = [2, 4, 8, 16]
            variations = []
            
            for scale in scales:
                if len(x) > scale:
                    var = 0
                    for i in range(0, len(x) - scale, scale):
                        dx = x[i + scale] - x[i]
                        dy = y[i + scale] - y[i]
                        var += np.sqrt(dx**2 + dy**2)
                    variations.append(var)
            
            if len(variations) > 1:
                # Fit log-log relationship
                log_scales = np.log(scales[:len(variations)])
                log_vars = np.log(variations)
                slope, _ = np.polyfit(log_scales, log_vars, 1)
                return 1 - slope
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _estimate_time_delay(self, data):
        """Estimate optimal time delay for phase space reconstruction."""
        try:
            # Use first minimum of autocorrelation
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first minimum
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                    return i
            return 1
        except:
            return 1
    
    def _phase_space_reconstruction(self, data, tau, dim):
        """Reconstruct phase space using time delay embedding."""
        n = len(data) - (dim - 1) * tau
        phase_space = np.zeros((n, dim))
        
        for i in range(dim):
            phase_space[:, i] = data[i*tau:i*tau + n]
            
        return phase_space

class AdaptiveFeatureSelector:
    """
    Dynamically select most informative features based on the data.
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        
    def select_features(self, features_dict, target_values=None, max_features=50):
        """
        Select most informative features using various criteria.
        """
        if target_values is not None:
            # Supervised feature selection
            return self._supervised_selection(features_dict, target_values, max_features)
        else:
            # Unsupervised feature selection based on variance
            return self._unsupervised_selection(features_dict, max_features)
    
    def _supervised_selection(self, features_dict, target_values, max_features):
        """Use mutual information for supervised feature selection."""
        from sklearn.feature_selection import mutual_info_regression
        
        feature_names = list(features_dict.keys())
        feature_matrix = np.array([[features_dict[name] for name in feature_names]])
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(feature_matrix.T, target_values)
        
        # Sort by importance
        sorted_indices = np.argsort(mi_scores)[::-1]
        
        selected_names = [feature_names[i] for i in sorted_indices[:max_features]]
        self.feature_importance = {name: score for name, score in zip(feature_names, mi_scores)}
        
        return {name: features_dict[name] for name in selected_names}
    
    def _unsupervised_selection(self, features_dict, max_features):
        """Select features based on variance and uniqueness."""
        # Calculate variance for each feature
        variances = {}
        for name, value in features_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                variances[name] = abs(value)  # Simple heuristic
        
        # Sort by variance
        sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        
        selected_names = [name for name, _ in sorted_features[:max_features]]
        return {name: features_dict[name] for name in selected_names}