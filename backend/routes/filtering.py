"""
Filtering API endpoints
Wraps desktop filtering code from src/filtering/
"""

from flask import Blueprint, request, jsonify, current_app
import numpy as np

# Import desktop filtering code
from src.filtering.filtering import (
    apply_savgol_filter,
    apply_fft_filter,
    butter_lowpass_filter,
    apply_wavelet_filter,
    combined_filter,
    calculate_filter_metrics
)

bp = Blueprint('filtering', __name__)


@bp.route('/savgol', methods=['POST'])
def apply_savgol():
    """
    Apply Savitzky-Golay filter
    
    Request JSON:
    {
        "data": [array of floats],
        "window_length": 51,
        "polyorder": 3
    }
    
    Returns:
    {
        "filtered_data": [array],
        "metrics": {
            "snr_improvement": float,
            "smoothness": float
        }
    }
    """
    try:
        data_dict = request.get_json()
        data = np.array(data_dict['data'])
        window_length = data_dict.get('window_length', 51)
        polyorder = data_dict.get('polyorder', 3)
        
        # Apply filter (REUSE desktop code!)
        filtered = apply_savgol_filter(data, window_length, polyorder)
        
        # Calculate metrics
        metrics = calculate_filter_metrics(data, filtered)
        
        return jsonify({
            'filtered_data': filtered.tolist(),
            'metrics': metrics
        })
        
    except Exception as e:
        current_app.logger.error(f"Savgol filter error: {str(e)}")
        return jsonify({'error': 'Filter failed', 'message': str(e)}), 500


@bp.route('/butterworth', methods=['POST'])
def apply_butterworth():
    """
    Apply Butterworth lowpass filter
    
    Request JSON:
    {
        "data": [array],
        "cutoff": 100,
        "fs": 1000,
        "order": 5
    }
    """
    try:
        data_dict = request.get_json()
        data = np.array(data_dict['data'])
        cutoff = data_dict.get('cutoff', 100)
        fs = data_dict.get('fs', 1000.0)
        order = data_dict.get('order', 5)
        
        filtered = butter_lowpass_filter(data, cutoff, fs, order)
        metrics = calculate_filter_metrics(data, filtered)
        
        return jsonify({
            'filtered_data': filtered.tolist(),
            'metrics': metrics
        })
        
    except Exception as e:
        current_app.logger.error(f"Butterworth filter error: {str(e)}")
        return jsonify({'error': 'Filter failed', 'message': str(e)}), 500


@bp.route('/wavelet', methods=['POST'])
def apply_wavelet():
    """
    Apply wavelet filter
    
    Request JSON:
    {
        "data": [array],
        "wavelet": "db4",
        "level": 3,
        "threshold_mode": "soft"
    }
    """
    try:
        data_dict = request.get_json()
        data = np.array(data_dict['data'])
        wavelet = data_dict.get('wavelet', 'db4')
        level = data_dict.get('level')
        threshold_mode = data_dict.get('threshold_mode', 'soft')
        
        filtered = apply_wavelet_filter(data, wavelet, level, threshold_mode)
        metrics = calculate_filter_metrics(data, filtered)
        
        return jsonify({
            'filtered_data': filtered.tolist(),
            'metrics': metrics
        })
        
    except Exception as e:
        current_app.logger.error(f"Wavelet filter error: {str(e)}")
        return jsonify({'error': 'Filter failed', 'message': str(e)}), 500


@bp.route('/combined', methods=['POST'])
def apply_combined():
    """
    Apply combined filters
    
    Request JSON:
    {
        "data": [array],
        "savgol_params": {...},
        "wavelet_params": {...},
        "butter_params": {...}
    }
    """
    try:
        data_dict = request.get_json()
        data = np.array(data_dict['data'])
        savgol_params = data_dict.get('savgol_params')
        wavelet_params = data_dict.get('wavelet_params')
        butter_params = data_dict.get('butter_params')
        
        filtered = combined_filter(
            data,
            savgol_params=savgol_params,
            wavelet_params=wavelet_params,
            butter_params=butter_params
        )
        
        metrics = calculate_filter_metrics(data, filtered)
        
        return jsonify({
            'filtered_data': filtered.tolist(),
            'metrics': metrics
        })
        
    except Exception as e:
        current_app.logger.error(f"Combined filter error: {str(e)}")
        return jsonify({'error': 'Filter failed', 'message': str(e)}), 500

