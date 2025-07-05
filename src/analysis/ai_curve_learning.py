# src/analysis/ai_curve_learning.py
"""
AI-powered curve learning and parameter extraction module.
Provides machine learning capabilities for action potential curve analysis,
automated parameter extraction, and intelligent curve fitting.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import curve_fit, differential_evolution
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from src.utils.logger import app_logger
except ImportError:
    import logging
    app_logger = logging.getLogger(__name__)

@dataclass
class CurveParameters:
    """Data class for storing curve fitting parameters."""
    amplitude: float
    time_constant: float
    offset: float
    linear_slope: float
    linear_intercept: float
    r_squared: float
    curve_type: str  # 'hyperpol' or 'depol'
    fitting_method: str
    confidence_score: float = 0.0

@dataclass
class TrainingDataPoint:
    """Data class for storing training data points."""
    time_data: np.ndarray
    current_data: np.ndarray
    parameters: CurveParameters
    metadata: Dict
    quality_score: float = 1.0

class CurveFeatureExtractor:
    """Extract features from curve data for machine learning."""
    
    def __init__(self):
        self.feature_names = [
            'peak_amplitude', 'time_to_peak', 'half_decay_time', 'area_under_curve',
            'initial_slope', 'final_slope', 'curve_length', 'noise_level',
            'skewness', 'kurtosis', 'zero_crossings', 'inflection_points',
            'early_phase_slope', 'late_phase_slope', 'transition_point',
            'baseline_drift', 'signal_to_noise_ratio', 'peak_width'
        ]
    
    def extract_features(self, time_data: np.ndarray, current_data: np.ndarray) -> np.ndarray:
        """Extract comprehensive features from curve data."""
        try:
            features = []
            
            # Basic amplitude features
            peak_amplitude = np.max(np.abs(current_data))
            features.append(peak_amplitude)
            
            # Time to peak
            peak_idx = np.argmax(np.abs(current_data))
            time_to_peak = time_data[peak_idx] if len(time_data) > peak_idx else 0
            features.append(time_to_peak)
            
            # Half decay time (time to reach half of peak value)
            half_amplitude = peak_amplitude / 2
            half_decay_idx = np.where(np.abs(current_data) <= half_amplitude)[0]
            half_decay_time = time_data[half_decay_idx[0]] if len(half_decay_idx) > 0 else time_data[-1]
            features.append(half_decay_time)
            
            # Area under curve
            area_under_curve = np.trapz(np.abs(current_data), time_data)
            features.append(area_under_curve)
            
            # Slope features
            if len(current_data) > 10:
                initial_slope = np.polyfit(time_data[:10], current_data[:10], 1)[0]
                final_slope = np.polyfit(time_data[-10:], current_data[-10:], 1)[0]
            else:
                initial_slope = final_slope = 0
            features.extend([initial_slope, final_slope])
            
            # Curve length
            curve_length = len(current_data)
            features.append(curve_length)
            
            # Noise level (standard deviation of high-frequency components)
            if len(current_data) > 5:
                smoothed = savgol_filter(current_data, min(11, len(current_data)//2*2+1), 3)
                noise_level = np.std(current_data - smoothed)
            else:
                noise_level = np.std(current_data)
            features.append(noise_level)
            
            # Statistical moments
            skewness = self._calculate_skewness(current_data)
            kurtosis = self._calculate_kurtosis(current_data)
            features.extend([skewness, kurtosis])
            
            # Zero crossings
            zero_crossings = len(np.where(np.diff(np.signbit(current_data)))[0])
            features.append(zero_crossings)
            
            # Inflection points (approximate)
            if len(current_data) > 3:
                second_derivative = np.gradient(np.gradient(current_data))
                inflection_points = len(np.where(np.diff(np.signbit(second_derivative)))[0])
            else:
                inflection_points = 0
            features.append(inflection_points)
            
            # Phase-specific slopes
            mid_point = len(current_data) // 2
            if mid_point > 5:
                early_phase_slope = np.polyfit(time_data[:mid_point], current_data[:mid_point], 1)[0]
                late_phase_slope = np.polyfit(time_data[mid_point:], current_data[mid_point:], 1)[0]
            else:
                early_phase_slope = late_phase_slope = 0
            features.extend([early_phase_slope, late_phase_slope])
            
            # Transition point (where slope changes most)
            if len(current_data) > 5:
                gradients = np.gradient(current_data)
                transition_point = np.argmax(np.abs(np.gradient(gradients)))
            else:
                transition_point = 0
            features.append(transition_point)
            
            # Baseline drift
            baseline_drift = current_data[-1] - current_data[0]
            features.append(baseline_drift)
            
            # Signal-to-noise ratio
            signal_power = np.mean(current_data**2)
            snr = signal_power / (noise_level**2) if noise_level > 0 else 1000
            features.append(snr)
            
            # Peak width (full width at half maximum)
            peak_width = self._calculate_peak_width(current_data, time_data)
            features.append(peak_width)
            
            return np.array(features)
            
        except Exception as e:
            app_logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(len(self.feature_names))
    
    def _calculate_skewness(self, data):
        """Calculate skewness of the data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of the data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _calculate_peak_width(self, data, time_data):
        """Calculate peak width at half maximum."""
        try:
            peak_val = np.max(np.abs(data))
            half_max = peak_val / 2
            
            # Find indices where signal is above half maximum
            above_half = np.where(np.abs(data) >= half_max)[0]
            
            if len(above_half) > 0:
                width_indices = above_half[-1] - above_half[0]
                if width_indices < len(time_data):
                    return time_data[width_indices] - time_data[0]
            
            return time_data[-1] - time_data[0]
            
        except Exception:
            return 0

class ExponentialFitter:
    """Advanced exponential curve fitting with multiple methods."""
    
    def __init__(self):
        self.fitting_methods = ['scipy_curve_fit', 'differential_evolution', 'least_squares']
    
    def fit_hyperpolarization(self, time_data: np.ndarray, current_data: np.ndarray, 
                            method: str = 'auto') -> CurveParameters:
        """Fit hyperpolarization curve: y = A * exp(-t/tau) + C + m*t + b"""
        
        def hyperpol_function(t, A, tau, C, m, b):
            return A * np.exp(-t / tau) + C + m * t + b
        
        return self._fit_curve(time_data, current_data, hyperpol_function, 
                              'hyperpol', method)
    
    def fit_depolarization(self, time_data: np.ndarray, current_data: np.ndarray,
                          method: str = 'auto') -> CurveParameters:
        """Fit depolarization curve: y = A * (1 - exp(-t/tau)) + C + m*t + b"""
        
        def depol_function(t, A, tau, C, m, b):
            return A * (1 - np.exp(-t / tau)) + C + m * t + b
        
        return self._fit_curve(time_data, current_data, depol_function, 
                              'depol', method)
    
    def _fit_curve(self, time_data: np.ndarray, current_data: np.ndarray, 
                   func, curve_type: str, method: str) -> CurveParameters:
        """Generic curve fitting with multiple fallback methods."""
        
        if method == 'auto':
            methods_to_try = self.fitting_methods
        else:
            methods_to_try = [method]
        
        best_params = None
        best_r2 = -np.inf
        best_method = None
        
        for fit_method in methods_to_try:
            try:
                params, r2 = self._try_fitting_method(time_data, current_data, 
                                                    func, fit_method)
                
                if r2 > best_r2:
                    best_params = params
                    best_r2 = r2
                    best_method = fit_method
                    
            except Exception as e:
                app_logger.debug(f"Fitting method {fit_method} failed: {str(e)}")
                continue
        
        if best_params is None:
            # Return default parameters if all methods fail
            return CurveParameters(
                amplitude=np.max(np.abs(current_data)),
                time_constant=0.025,  # 25 ms default
                offset=np.mean(current_data),
                linear_slope=0,
                linear_intercept=0,
                r_squared=0.0,
                curve_type=curve_type,
                fitting_method='failed',
                confidence_score=0.0
            )
        
        # Calculate confidence score based on R²
        confidence_score = min(best_r2, 1.0) if best_r2 > 0 else 0.0
        
        return CurveParameters(
            amplitude=best_params[0],
            time_constant=abs(best_params[1]),  # Ensure positive
            offset=best_params[2],
            linear_slope=best_params[3],
            linear_intercept=best_params[4],
            r_squared=best_r2,
            curve_type=curve_type,
            fitting_method=best_method,
            confidence_score=confidence_score
        )
    
    def _try_fitting_method(self, time_data: np.ndarray, current_data: np.ndarray,
                           func, method: str) -> Tuple[np.ndarray, float]:
        """Try a specific fitting method."""
        
        if method == 'scipy_curve_fit':
            return self._scipy_curve_fit(time_data, current_data, func)
        elif method == 'differential_evolution':
            return self._differential_evolution_fit(time_data, current_data, func)
        elif method == 'least_squares':
            return self._least_squares_fit(time_data, current_data, func)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
    
    def _scipy_curve_fit(self, time_data: np.ndarray, current_data: np.ndarray, 
                        func) -> Tuple[np.ndarray, float]:
        """Standard scipy curve_fit method."""
        
        # Initial parameter guesses
        A_guess = np.max(np.abs(current_data)) - np.min(np.abs(current_data))
        tau_guess = (time_data[-1] - time_data[0]) / 3  # 1/3 of time range
        C_guess = np.mean(current_data)
        m_guess = (current_data[-1] - current_data[0]) / (time_data[-1] - time_data[0])
        b_guess = current_data[0]
        
        initial_guess = [A_guess, tau_guess, C_guess, m_guess, b_guess]
        
        # Parameter bounds
        bounds = (
            [-np.inf, 0.001, -np.inf, -np.inf, -np.inf],  # Lower bounds
            [np.inf, 10.0, np.inf, np.inf, np.inf]         # Upper bounds
        )
        
        popt, _ = curve_fit(func, time_data, current_data, 
                           p0=initial_guess, bounds=bounds, 
                           maxfev=5000)
        
        # Calculate R²
        y_pred = func(time_data, *popt)
        r2 = r2_score(current_data, y_pred)
        
        return popt, r2
    
    def _differential_evolution_fit(self, time_data: np.ndarray, current_data: np.ndarray,
                                   func) -> Tuple[np.ndarray, float]:
        """Differential evolution global optimization."""
        
        def objective(params):
            try:
                y_pred = func(time_data, *params)
                return np.sum((current_data - y_pred) ** 2)
            except:
                return 1e10
        
        # Parameter bounds for differential evolution
        amplitude_bound = max(np.max(np.abs(current_data)) * 2, 1000)
        bounds = [
            (-amplitude_bound, amplitude_bound),  # A
            (0.001, 1.0),                        # tau
            (-amplitude_bound, amplitude_bound),  # C
            (-1000, 1000),                       # m
            (-amplitude_bound, amplitude_bound)   # b
        ]
        
        result = differential_evolution(objective, bounds, maxiter=1000, seed=42)
        
        if result.success:
            y_pred = func(time_data, *result.x)
            r2 = r2_score(current_data, y_pred)
            return result.x, r2
        else:
            raise RuntimeError("Differential evolution failed to converge")
    
    def _least_squares_fit(self, time_data: np.ndarray, current_data: np.ndarray,
                          func) -> Tuple[np.ndarray, float]:
        """Simple least squares with linearization."""
        
        # For exponential curves, try log-linear fitting first
        try:
            # Remove offset estimate
            offset_est = np.mean(current_data[-10:])  # Estimate from last points
            data_corrected = current_data - offset_est
            
            # Take log of absolute values
            log_data = np.log(np.abs(data_corrected) + 1e-10)
            
            # Linear fit in log space
            coeffs = np.polyfit(time_data, log_data, 1)
            
            # Convert back to exponential parameters
            tau_est = -1 / coeffs[0] if coeffs[0] != 0 else 0.025
            A_est = np.exp(coeffs[1])
            
            # Refine with full nonlinear fit
            initial_guess = [A_est, abs(tau_est), offset_est, 0, 0]
            
            popt, _ = curve_fit(func, time_data, current_data, 
                               p0=initial_guess, maxfev=2000)
            
            y_pred = func(time_data, *popt)
            r2 = r2_score(current_data, y_pred)
            
            return popt, r2
            
        except Exception:
            # Fallback to simple linear regression
            A_guess = np.max(np.abs(current_data))
            tau_guess = 0.025
            C_guess = np.mean(current_data)
            m_guess = 0
            b_guess = 0
            
            params = np.array([A_guess, tau_guess, C_guess, m_guess, b_guess])
            y_pred = func(time_data, *params)
            r2 = r2_score(current_data, y_pred)
            
            return params, r2

class CurveLearningModel:
    """Machine learning model for curve parameter prediction."""
    
    def __init__(self, model_type: str = 'random_forest'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for machine learning features")
        
        self.model_type = model_type
        self.feature_extractor = CurveFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}  # Separate models for different parameters
        self.is_trained = False
        
        # Initialize models for different parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different parameters."""
        
        model_configs = {
            'random_forest': {
                'amplitude': RandomForestRegressor(n_estimators=100, random_state=42),
                'time_constant': RandomForestRegressor(n_estimators=100, random_state=42),
                'offset': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_slope': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_intercept': RandomForestRegressor(n_estimators=100, random_state=42)
            },
            'gradient_boosting': {
                'amplitude': GradientBoostingRegressor(random_state=42),
                'time_constant': GradientBoostingRegressor(random_state=42),
                'offset': GradientBoostingRegressor(random_state=42),
                'linear_slope': GradientBoostingRegressor(random_state=42),
                'linear_intercept': GradientBoostingRegressor(random_state=42)
            },
            'neural_network': {
                'amplitude': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
                'time_constant': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
                'offset': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
                'linear_slope': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
                'linear_intercept': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
            }
        }
        
        self.models = model_configs.get(self.model_type, model_configs['random_forest'])
    
    def train(self, training_data: List[TrainingDataPoint]) -> Dict[str, float]:
        """Train the model on labeled curve data."""
        
        if len(training_data) < 5:
            raise ValueError("Need at least 5 training examples")
        
        app_logger.info(f"Training curve learning model with {len(training_data)} examples")
        
        # Extract features and labels
        features = []
        labels = {
            'amplitude': [],
            'time_constant': [],
            'offset': [],
            'linear_slope': [],
            'linear_intercept': []
        }
        
        for data_point in training_data:
            # Extract features
            curve_features = self.feature_extractor.extract_features(
                data_point.time_data, data_point.current_data
            )
            features.append(curve_features)
            
            # Extract labels
            labels['amplitude'].append(data_point.parameters.amplitude)
            labels['time_constant'].append(data_point.parameters.time_constant)
            labels['offset'].append(data_point.parameters.offset)
            labels['linear_slope'].append(data_point.parameters.linear_slope)
            labels['linear_intercept'].append(data_point.parameters.linear_intercept)
        
        features = np.array(features)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models for each parameter
        training_scores = {}
        
        for param_name, param_values in labels.items():
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, param_values, test_size=0.2, random_state=42
                )
                
                # Train model
                self.models[param_name].fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.models[param_name].predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                training_scores[param_name] = {
                    'r2': r2,
                    'rmse': rmse
                }
                
                app_logger.info(f"{param_name} model - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                
            except Exception as e:
                app_logger.error(f"Failed to train {param_name} model: {str(e)}")
                training_scores[param_name] = {'r2': 0.0, 'rmse': float('inf')}
        
        self.is_trained = True
        return training_scores
    
    def predict(self, time_data: np.ndarray, current_data: np.ndarray, 
                curve_type: str = 'hyperpol') -> CurveParameters:
        """Predict curve parameters using the trained model."""
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Extract features
        features = self.feature_extractor.extract_features(time_data, current_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict parameters
        predictions = {}
        confidences = {}
        
        for param_name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)[0]
                predictions[param_name] = pred
                
                # Estimate confidence (simplified)
                if hasattr(model, 'predict_proba'):
                    # For models that support probability prediction
                    confidences[param_name] = 0.8  # Placeholder
                else:
                    confidences[param_name] = 0.7  # Default confidence
                    
            except Exception as e:
                app_logger.error(f"Prediction failed for {param_name}: {str(e)}")
                predictions[param_name] = 0.0
                confidences[param_name] = 0.0
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(confidences.values()))
        
        return CurveParameters(
            amplitude=predictions['amplitude'],
            time_constant=max(predictions['time_constant'], 0.001),  # Ensure positive
            offset=predictions['offset'],
            linear_slope=predictions['linear_slope'],
            linear_intercept=predictions['linear_intercept'],
            r_squared=overall_confidence,  # Use confidence as proxy
            curve_type=curve_type,
            fitting_method='ml_prediction',
            confidence_score=overall_confidence
        )
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_extractor.feature_names,
            'is_trained': self.is_trained,
            'version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        app_logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        app_logger.info(f"Model loaded from {filepath}")

class AICurveLearning:
    """Main AI curve learning interface."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.exponential_fitter = ExponentialFitter()
        self.training_data = []
        
        # Initialize ML model if sklearn is available
        if SKLEARN_AVAILABLE:
            self.ml_model = CurveLearningModel(model_type)
        else:
            self.ml_model = None
            app_logger.warning("scikit-learn not available - ML features disabled")
        
        # Model storage directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def analyze_curve(self, time_data: np.ndarray, current_data: np.ndarray, 
                     curve_type: str = 'auto', method: str = 'auto') -> CurveParameters:
        """Analyze a curve and extract parameters using the best available method."""
        
        # Auto-detect curve type if not specified
        if curve_type == 'auto':
            curve_type = self._detect_curve_type(current_data)
        
        # Try ML prediction first if model is trained
        if (self.ml_model and self.ml_model.is_trained and 
            method in ['auto', 'ml']):
            try:
                ml_params = self.ml_model.predict(time_data, current_data, curve_type)
                if ml_params.confidence_score > 0.7:  # High confidence threshold
                    app_logger.info(f"Using ML prediction (confidence: {ml_params.confidence_score:.2f})")
                    return ml_params
            except Exception as e:
                app_logger.warning(f"ML prediction failed, falling back to curve fitting: {str(e)}")
        
        # Fall back to traditional curve fitting
        if curve_type == 'hyperpol':
            return self.exponential_fitter.fit_hyperpolarization(time_data, current_data, method)
        else:
            return self.exponential_fitter.fit_depolarization(time_data, current_data, method)
    
    def _detect_curve_type(self, current_data: np.ndarray) -> str:
        """Auto-detect if curve is hyperpolarization or depolarization."""
        
        # Simple heuristic based on curve shape
        peak_idx = np.argmax(np.abs(current_data))
        
        if peak_idx < len(current_data) // 2:
            # Peak is in first half - likely hyperpolarization (rapid rise, slow decay)
            return 'hyperpol'
        else:
            # Peak is in second half - likely depolarization (slow rise)
            return 'depol'
    
    def add_training_data(self, time_data: np.ndarray, current_data: np.ndarray,
                         manual_parameters: CurveParameters, 
                         metadata: Dict = None) -> bool:
        """Add manually validated data for training."""
        
        if metadata is None:
            metadata = {}
        
        # Calculate quality score based on R²
        quality_score = max(manual_parameters.r_squared, 0.5)
        
        training_point = TrainingDataPoint(
            time_data=time_data.copy(),
            current_data=current_data.copy(),
            parameters=manual_parameters,
            metadata=metadata,
            quality_score=quality_score
        )
        
        self.training_data.append(training_point)
        
        app_logger.info(f"Added training data point (total: {len(self.training_data)})")
        return True
    
    def train_model(self) -> Dict[str, float]:
        """Train the ML model on accumulated training data."""
        
        if not self.ml_model:
            raise RuntimeError("ML model not available (scikit-learn not installed)")
        
        if len(self.training_data) < 5:
            raise ValueError(f"Need at least 5 training examples, have {len(self.training_data)}")
        
        # Filter training data by quality
        high_quality_data = [
            point for point in self.training_data 
            if point.quality_score > 0.7
        ]
        
        if len(high_quality_data) < 3:
            app_logger.warning("Using all training data due to insufficient high-quality examples")
            training_data = self.training_data
        else:
            training_data = high_quality_data
        
        # Train the model
        training_scores = self.ml_model.train(training_data)
        
        # Auto-save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"curve_model_{timestamp}.pkl"
        self.ml_model.save_model(str(model_path))
        
        return training_scores
    
    def load_pretrained_model(self, model_path: str = None) -> bool:
        """Load a pre-trained model."""
        
        if not self.ml_model:
            app_logger.error("ML model not available")
            return False
        
        if model_path is None:
            # Find the most recent model
            model_files = list(self.model_dir.glob("curve_model_*.pkl"))
            if not model_files:
                app_logger.warning("No pre-trained models found")
                return False
            
            model_path = max(model_files, key=os.path.getctime)
            app_logger.info(f"Loading most recent model: {model_path}")
        
        try:
            self.ml_model.load_model(str(model_path))
            return True
        except Exception as e:
            app_logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def export_training_data(self, filepath: str):
        """Export training data for external analysis."""
        
        export_data = []
        
        for point in self.training_data:
            data_dict = {
                'time_data': point.time_data.tolist(),
                'current_data': point.current_data.tolist(),
                'amplitude': point.parameters.amplitude,
                'time_constant': point.parameters.time_constant,
                'offset': point.parameters.offset,
                'linear_slope': point.parameters.linear_slope,
                'linear_intercept': point.parameters.linear_intercept,
                'r_squared': point.parameters.r_squared,
                'curve_type': point.parameters.curve_type,
                'fitting_method': point.parameters.fitting_method,
                'quality_score': point.quality_score,
                'metadata': point.metadata
            }
            export_data.append(data_dict)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        app_logger.info(f"Training data exported to {filepath}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics of the current model."""
        
        if not self.ml_model or not self.ml_model.is_trained:
            return {'status': 'no_model', 'message': 'No trained model available'}
        
        # Test on training data (for demonstration)
        test_results = []
        
        for point in self.training_data[-5:]:  # Test on last 5 points
            try:
                predicted = self.ml_model.predict(
                    point.time_data, point.current_data, point.parameters.curve_type
                )
                
                actual = point.parameters
                
                # Calculate relative errors
                amp_error = abs(predicted.amplitude - actual.amplitude) / abs(actual.amplitude)
                tau_error = abs(predicted.time_constant - actual.time_constant) / actual.time_constant
                
                test_results.append({
                    'amplitude_error': amp_error,
                    'time_constant_error': tau_error,
                    'confidence': predicted.confidence_score
                })
                
            except Exception as e:
                app_logger.error(f"Error testing model performance: {str(e)}")
        
        if test_results:
            avg_amp_error = np.mean([r['amplitude_error'] for r in test_results])
            avg_tau_error = np.mean([r['time_constant_error'] for r in test_results])
            avg_confidence = np.mean([r['confidence'] for r in test_results])
            
            return {
                'status': 'available',
                'training_samples': len(self.training_data),
                'avg_amplitude_error': avg_amp_error,
                'avg_time_constant_error': avg_tau_error,
                'avg_confidence': avg_confidence,
                'model_type': self.ml_model.model_type
            }
        else:
            return {
                'status': 'available',
                'training_samples': len(self.training_data),
                'model_type': self.ml_model.model_type,
                'message': 'No test results available'
            }

# Convenience function for easy integration
def create_ai_curve_analyzer(model_type: str = 'random_forest') -> AICurveLearning:
    """Create and return an AI curve analyzer instance."""
    return AICurveLearning(model_type)

# Module-level constants
SUPPORTED_CURVE_TYPES = ['hyperpol', 'depol', 'auto']
SUPPORTED_FITTING_METHODS = ['auto', 'scipy_curve_fit', 'differential_evolution', 
                           'least_squares', 'ml']
SUPPORTED_MODEL_TYPES = ['random_forest', 'gradient_boosting', 'neural_network']

def get_module_info() -> Dict:
    """Get information about the AI curve learning module."""
    return {
        'sklearn_available': SKLEARN_AVAILABLE,
        'scipy_available': SCIPY_AVAILABLE,
        'supported_curve_types': SUPPORTED_CURVE_TYPES,
        'supported_fitting_methods': SUPPORTED_FITTING_METHODS,
        'supported_model_types': SUPPORTED_MODEL_TYPES,
        'version': '1.0.0'
    }