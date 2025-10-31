"""
Machine Learning Models for Excel Learning

This module contains various ML models for learning Excel patterns and generating data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    parameters: Dict[str, Any]
    input_features: List[str]
    output_features: List[str]
    data_type: str  # 'numeric', 'categorical', 'mixed'

class ExcelMLModels:
    """
    Collection of ML models for Excel data learning and generation
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.configs = {}
        
        # Initialize TensorFlow
        tf.random.set_seed(42)
        
    def create_sequential_model(self, input_dim: int, output_dim: int, 
                              layers: List[int] = [64, 32, 16]) -> keras.Model:
        """Create a sequential neural network model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(layers[0], activation='relu', input_shape=(input_dim,)))
        model.add(keras.layers.Dropout(0.2))
        
        # Hidden layers
        for units in layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
        
        # Output layer
        model.add(keras.layers.Dense(output_dim, activation='linear'))
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_lstm_model(self, sequence_length: int, features: int, 
                         output_dim: int) -> keras.Model:
        """Create an LSTM model for time series data"""
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(25),
            keras.layers.Dense(output_dim)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_numeric_model(self, data: pd.DataFrame, target_column: str, 
                           model_name: str, model_type: str = 'neural_network') -> Dict[str, Any]:
        """
        Train a model for numeric data generation
        
        Args:
            data: Training data
            target_column: Column to predict
            model_name: Name for the model
            model_type: Type of model ('neural_network', 'random_forest', 'linear')
            
        Returns:
            Training results
        """
        logger.info(f"Training {model_type} model for {target_column}")
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        X_encoded, encoders = self._encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'neural_network':
            model = self.create_sequential_model(
                input_dim=X_train_scaled.shape[1], 
                output_dim=1
            )
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=100,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test_scaled).flatten()
            mse = mean_squared_error(y_test, y_pred)
            
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            history = None
            
        elif model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            history = None
        
        # Save model and components
        self._save_model_components(model_name, model, scaler, encoders, model_type)
        
        # Store configuration
        config = ModelConfig(
            model_type=model_type,
            parameters={'input_dim': X_train_scaled.shape[1], 'output_dim': 1},
            input_features=list(X.columns),
            output_features=[target_column],
            data_type='numeric'
        )
        self.configs[model_name] = config
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'feature_importance': self._get_feature_importance(model, X.columns) if hasattr(model, 'feature_importances_') else None,
            'history': history.history if history else None
        }
        
        logger.info(f"Model {model_name} trained successfully. RMSE: {results['rmse']:.4f}")
        return results
    
    def train_categorical_model(self, data: pd.DataFrame, target_column: str,
                               model_name: str, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train a model for categorical data generation
        
        Args:
            data: Training data
            target_column: Column to predict
            model_name: Name for the model
            model_type: Type of model ('random_forest', 'neural_network', 'logistic')
            
        Returns:
            Training results
        """
        logger.info(f"Training {model_type} model for categorical {target_column}")
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Handle categorical features
        X_encoded, encoders = self._encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
        elif model_type == 'neural_network':
            num_classes = len(label_encoder.classes_)
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            y_pred = model.predict(X_test_scaled).argmax(axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
        
        # Save model and components
        self._save_model_components(model_name, model, scaler, encoders, model_type, label_encoder)
        
        # Store configuration
        config = ModelConfig(
            model_type=model_type,
            parameters={'input_dim': X_train_scaled.shape[1], 'num_classes': len(label_encoder.classes_)},
            input_features=list(X.columns),
            output_features=[target_column],
            data_type='categorical'
        )
        self.configs[model_name] = config
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'accuracy': accuracy,
            'feature_importance': self._get_feature_importance(model, X.columns) if hasattr(model, 'feature_importances_') else None,
            'classes': label_encoder.classes_.tolist()
        }
        
        logger.info(f"Model {model_name} trained successfully. Accuracy: {accuracy:.4f}")
        return results
    
    def train_pattern_model(self, data: pd.DataFrame, pattern_type: str,
                           model_name: str) -> Dict[str, Any]:
        """
        Train a model for specific data patterns
        
        Args:
            data: Training data
            pattern_type: Type of pattern ('sequential', 'formula', 'random')
            model_name: Name for the model
            
        Returns:
            Training results
        """
        logger.info(f"Training pattern model for {pattern_type} pattern")
        
        if pattern_type == 'sequential':
            return self._train_sequential_pattern_model(data, model_name)
        elif pattern_type == 'formula':
            return self._train_formula_pattern_model(data, model_name)
        elif pattern_type == 'random':
            return self._train_random_pattern_model(data, model_name)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
    
    def _train_sequential_pattern_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Train model for sequential patterns"""
        # For sequential patterns, we'll use LSTM or simple regression
        if len(data) > 10:
            # Use LSTM for longer sequences
            sequence_length = min(5, len(data) // 2)
            X, y = self._create_sequences(data, sequence_length)
            
            model = self.create_lstm_model(
                sequence_length=sequence_length,
                features=X.shape[2],
                output_dim=y.shape[1]
            )
            
            history = model.fit(X, y, epochs=50, batch_size=16, verbose=0)
            
            # Save model
            model_path = self.models_dir / f"{model_name}.h5"
            model.save(model_path)
            
            return {
                'model_name': model_name,
                'model_type': 'lstm',
                'sequence_length': sequence_length,
                'history': history.history
            }
        else:
            # Use simple linear regression for short sequences
            X = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Save model
            model_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            return {
                'model_name': model_name,
                'model_type': 'linear',
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_.tolist()
            }
    
    def _train_formula_pattern_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Train model for formula patterns"""
        # Try to identify the formula pattern
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        # Try polynomial regression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        
        poly_model.fit(X, y)
        
        # Save model
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(poly_model, model_path)
        
        return {
            'model_name': model_name,
            'model_type': 'polynomial',
            'degree': 3
        }
    
    def _train_random_pattern_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Train model for random patterns"""
        # For random patterns, we'll use statistical models
        stats = {
            'mean': data.mean().tolist(),
            'std': data.std().tolist(),
            'min': data.min().tolist(),
            'max': data.max().tolist(),
            'distribution': 'normal'  # Assume normal distribution
        }
        
        # Save statistics
        stats_path = self.models_dir / f"{model_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        return {
            'model_name': model_name,
            'model_type': 'statistical',
            'statistics': stats
        }
    
    def generate_data(self, model_name: str, num_samples: int, 
                     input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate data using a trained model
        
        Args:
            model_name: Name of the trained model
            num_samples: Number of samples to generate
            input_data: Optional input data for conditional generation
            
        Returns:
            Generated data
        """
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.configs[model_name]
        
        if config.data_type == 'numeric':
            return self._generate_numeric_data(model_name, num_samples, input_data)
        elif config.data_type == 'categorical':
            return self._generate_categorical_data(model_name, num_samples, input_data)
        else:
            raise ValueError(f"Unsupported data type: {config.data_type}")
    
    def _generate_numeric_data(self, model_name: str, num_samples: int,
                              input_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Generate numeric data"""
        # Load model
        model_path = self.models_dir / f"{model_name}.h5"
        if model_path.exists():
            model = keras.models.load_model(model_path)
            model_type = 'neural_network'
        else:
            model_path = self.models_dir / f"{model_name}.pkl"
            model = joblib.load(model_path)
            model_type = 'traditional'
        
        # Load scaler and encoders
        scaler = joblib.load(self.models_dir / f"{model_name}_scaler.pkl")
        encoders = joblib.load(self.models_dir / f"{model_name}_encoders.pkl")
        
        # Prepare input data
        if input_data is None:
            # Generate random input data
            config = self.configs[model_name]
            input_features = config.input_features
            input_data = pd.DataFrame(
                np.random.randn(num_samples, len(input_features)),
                columns=input_features
            )
        
        # Encode categorical features
        input_encoded = self._encode_features(input_data, encoders)
        
        # Scale features
        input_scaled = scaler.transform(input_encoded)
        
        # Generate predictions
        if model_type == 'neural_network':
            predictions = model.predict(input_scaled)
        else:
            predictions = model.predict(input_scaled)
        
        # Create output DataFrame
        output_data = input_data.copy()
        output_data[self.configs[model_name].output_features[0]] = predictions.flatten()
        
        return output_data
    
    def _generate_categorical_data(self, model_name: str, num_samples: int,
                                  input_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Generate categorical data"""
        # Load model
        model_path = self.models_dir / f"{model_name}.h5"
        if model_path.exists():
            model = keras.models.load_model(model_path)
            model_type = 'neural_network'
        else:
            model_path = self.models_dir / f"{model_name}.pkl"
            model = joblib.load(model_path)
            model_type = 'traditional'
        
        # Load components
        scaler = joblib.load(self.models_dir / f"{model_name}_scaler.pkl")
        encoders = joblib.load(self.models_dir / f"{model_name}_encoders.pkl")
        label_encoder = joblib.load(self.models_dir / f"{model_name}_label_encoder.pkl")
        
        # Prepare input data
        if input_data is None:
            config = self.configs[model_name]
            input_features = config.input_features
            input_data = pd.DataFrame(
                np.random.randn(num_samples, len(input_features)),
                columns=input_features
            )
        
        # Encode and scale input
        input_encoded = self._encode_features(input_data, encoders)
        input_scaled = scaler.transform(input_encoded)
        
        # Generate predictions
        if model_type == 'neural_network':
            predictions_proba = model.predict(input_scaled)
            predictions = predictions_proba.argmax(axis=1)
        else:
            predictions = model.predict(input_scaled)
        
        # Decode predictions
        predictions_decoded = label_encoder.inverse_transform(predictions)
        
        # Create output DataFrame
        output_data = input_data.copy()
        output_data[self.configs[model_name].output_features[0]] = predictions_decoded
        
        return output_data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Encode categorical features"""
        encoded_data = data.copy()
        encoders = {}
        
        for column in data.columns:
            if data[column].dtype == 'object':
                encoder = LabelEncoder()
                encoded_data[column] = encoder.fit_transform(data[column].astype(str))
                encoders[column] = encoder
        
        return encoded_data, encoders
    
    def _encode_features(self, data: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
        """Encode features using pre-trained encoders"""
        encoded_data = data.copy()
        
        for column, encoder in encoders.items():
            if column in data.columns:
                encoded_data[column] = encoder.transform(data[column].astype(str))
        
        return encoded_data
    
    def _create_sequences(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:i+sequence_length].values)
            y.append(data.iloc[i+sequence_length].values)
        
        return np.array(X), np.array(y)
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_)))
        else:
            return {}
    
    def _save_model_components(self, model_name: str, model, scaler, encoders, 
                              model_type: str, label_encoder=None):
        """Save model and its components"""
        # Save model
        if model_type == 'neural_network':
            model_path = self.models_dir / f"{model_name}.h5"
            model.save(model_path)
        else:
            model_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Save encoders
        encoders_path = self.models_dir / f"{model_name}_encoders.pkl"
        joblib.dump(encoders, encoders_path)
        
        # Save label encoder if provided
        if label_encoder is not None:
            label_encoder_path = self.models_dir / f"{model_name}_label_encoder.pkl"
            joblib.dump(label_encoder, label_encoder_path)
        
        # Save config
        config_path = self.models_dir / f"{model_name}_config.json"
        config = self.configs[model_name]
        config_dict = {
            'model_type': config.model_type,
            'parameters': config.parameters,
            'input_features': config.input_features,
            'output_features': config.output_features,
            'data_type': config.data_type
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_model(self, model_name: str):
        """Load a trained model"""
        config_path = self.models_dir / f"{model_name}_config.json"
        if not config_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(**config_dict)
        self.configs[model_name] = config
        
        logger.info(f"Model {model_name} loaded successfully")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        config_files = list(self.models_dir.glob("*_config.json"))
        return [f.stem.replace("_config", "") for f in config_files]
    
    def delete_model(self, model_name: str):
        """Delete a model and its components"""
        files_to_delete = [
            f"{model_name}.h5",
            f"{model_name}.pkl",
            f"{model_name}_scaler.pkl",
            f"{model_name}_encoders.pkl",
            f"{model_name}_label_encoder.pkl",
            f"{model_name}_config.json",
            f"{model_name}_stats.json"
        ]
        
        for filename in files_to_delete:
            file_path = self.models_dir / filename
            if file_path.exists():
                file_path.unlink()
        
        if model_name in self.configs:
            del self.configs[model_name]
        
        logger.info(f"Model {model_name} deleted successfully")
