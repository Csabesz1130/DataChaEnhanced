"""
Analysis API endpoints
Wraps desktop analysis code from src/analysis/
"""

from flask import Blueprint, request, jsonify, current_app
import numpy as np
import sys
from pathlib import Path

# Import desktop analysis code
from src.analysis.action_potential import ActionPotentialProcessor
from backend.utils.file_handler import get_file_data
from backend.utils.plot_converter import convert_matplotlib_to_json
from backend.utils.db import db, AnalysisResult

bp = Blueprint('analysis', __name__)


@bp.route('/process', methods=['POST'])
def process_signal():
    """
    Process signal with action potential analysis
    
    Request JSON:
    {
        "file_id": "uuid-string",
        "params": {
            "n_cycles": 2,
            "t0": 20,
            "t1": 100,
            ... etc
        },
        "options": {
            "use_alternative_method": false,
            "auto_optimize_starting_point": true
        }
    }
    
    Returns:
    {
        "analysis_id": "uuid-string",
        "results": {
            "orange_curve": [...],
            "normalized_curve": [...],
            "average_curve": [...],
            "modified_hyperpol": [...],
            "modified_depol": [...],
            "cycles": [...],
            "baseline": float
        },
        "metadata": {
            "processing_time": float,
            "params_used": {...}
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()
        file_id = data.get('file_id')
        params = data.get('params', {})
        options = data.get('options', {})
        
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Get file data
        current_app.logger.info(f"Processing signal for file_id: {file_id}")
        signal_data, time_data, file_info = get_file_data(file_id)
        
        # Ensure data is numpy array and copy it (like desktop app's filtered_data)
        import numpy as np
        filtered_data = np.asarray(signal_data, dtype=np.float64).copy()
        time_data = np.asarray(time_data, dtype=np.float64)
        
        # Merge with default params
        default_params = current_app.config['DEFAULT_ANALYSIS_PARAMS'].copy()
        default_params.update(params)
        
        # Ensure 'n' parameter is handled correctly (convert to int if provided)
        if 'n' in default_params and default_params['n'] is not None:
            default_params['n'] = int(default_params['n'])
        
        # Create processor (REUSE desktop code!) - use filtered_data like desktop app
        processor = ActionPotentialProcessor(
            filtered_data, 
            time_data, 
            default_params
        )
        
        # Process signal with options (exactly like desktop app)
        import time
        start_time = time.time()
        
        # Call process_signal and capture return values (like desktop app does)
        (
            processed_data,
            orange_curve,
            orange_times,
            normalized_curve,
            normalized_times,
            average_curve,
            average_times,
            results
        ) = processor.process_signal(
            use_alternative_method=options.get('use_alternative_method', False),
            auto_optimize_starting_point=options.get('auto_optimize_starting_point', True)
        )
        
        # Check for pipeline failure (like desktop app)
        if processed_data is None:
            error_msg = results.get('error', 'Unknown error') if isinstance(results, dict) else 'Processing failed'
            current_app.logger.error(f"Analysis pipeline failed: {error_msg}")
            return jsonify({'error': 'Analysis failed', 'message': error_msg}), 500
        
        processing_time = time.time() - start_time
        
        # Extract results using the return values from process_signal (like desktop app)
        # Also access processor attributes to ensure we have all curves
        results_dict = {
            'orange_curve': orange_curve.tolist() if orange_curve is not None else None,
            'orange_curve_times': orange_times.tolist() if orange_times is not None else None,
            
            'normalized_curve': normalized_curve.tolist() if normalized_curve is not None else None,
            'normalized_curve_times': normalized_times.tolist() if normalized_times is not None else None,
            
            'average_curve': average_curve.tolist() if average_curve is not None else None,
            'average_curve_times': average_times.tolist() if average_times is not None else None,
            
            # Purple curves are generated in process_signal, access from processor
            'modified_hyperpol': processor.modified_hyperpol.tolist() if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None else None,
            'modified_hyperpol_times': processor.modified_hyperpol_times.tolist() if hasattr(processor, 'modified_hyperpol_times') and processor.modified_hyperpol_times is not None else None,
            
            'modified_depol': processor.modified_depol.tolist() if hasattr(processor, 'modified_depol') and processor.modified_depol is not None else None,
            'modified_depol_times': processor.modified_depol_times.tolist() if hasattr(processor, 'modified_depol_times') and processor.modified_depol_times is not None else None,
            
            'baseline': float(processor.baseline) if hasattr(processor, 'baseline') and processor.baseline is not None else None,
            
            'cycles': len(processor.cycles) if hasattr(processor, 'cycles') and processor.cycles else 0,
            
            # Include processed_data for reference
            'processed_data': processed_data.tolist() if processed_data is not None else None
        }
        
        # Save to database
        analysis_result = AnalysisResult(
            file_id=file_id,
            params=default_params,
            results=results_dict,
            processing_time=processing_time
        )
        db.session.add(analysis_result)
        db.session.commit()
        
        current_app.logger.info(f"Analysis completed in {processing_time:.2f}s, ID: {analysis_result.id}")
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'results': results_dict,
            'metadata': {
                'processing_time': processing_time,
                'params_used': default_params,
                'file_name': file_info['filename']
            }
        })
        
    except ValueError as e:
        current_app.logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': 'Invalid input', 'message': str(e)}), 400
        
    except Exception as e:
        current_app.logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Analysis failed', 'message': str(e)}), 500


@bp.route('/integrals', methods=['POST'])
def calculate_integrals():
    """
    Calculate integrals for hyperpolarization and depolarization curves
    
    Request JSON:
    {
        "analysis_id": "uuid-string",
        "ranges": {
            "hyperpol": {"start": 0, "end": 199},
            "depol": {"start": 0, "end": 199}
        },
        "method": "direct",  // or "linreg"
        "linreg_params": {  // optional, for linreg method
            "hyperpol": {"slope": 0, "intercept": 0},
            "depol": {"slope": 0, "intercept": 0}
        }
    }
    
    Returns:
    {
        "hyperpol_integral": float,
        "depol_integral": float,
        "capacitance": float,
        "method_used": "direct"
    }
    """
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        ranges = data.get('ranges', {
            'hyperpol': {'start': 0, 'end': 199},
            'depol': {'start': 0, 'end': 199}
        })
        method = data.get('method', 'direct')
        linreg_params = data.get('linreg_params')
        
        if not analysis_id:
            return jsonify({'error': 'analysis_id is required'}), 400
        
        # Get analysis results from database
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Reconstruct processor from saved results
        # (In practice, might want to recreate from original file)
        results = analysis.results
        
        # Convert lists back to numpy arrays
        modified_hyperpol = np.array(results['modified_hyperpol'])
        modified_hyperpol_times = np.array(results['modified_hyperpol_times'])
        modified_depol = np.array(results['modified_depol'])
        modified_depol_times = np.array(results['modified_depol_times'])
        
        # Create a minimal processor to use integrate_segment method
        # Or implement integration logic here directly
        from scipy import integrate
        
        # Extract ranges
        hyperpol_range = ranges['hyperpol']
        h_start, h_end = hyperpol_range['start'], hyperpol_range['end']
        
        depol_range = ranges['depol']
        d_start, d_end = depol_range['start'], depol_range['end']
        
        # Calculate integrals using trapezoidal rule
        hyperpol_data = modified_hyperpol[h_start:h_end+1]
        hyperpol_times = modified_hyperpol_times[h_start:h_end+1]
        hyperpol_integral = np.abs(integrate.trapz(hyperpol_data, hyperpol_times * 1000))
        
        depol_data = modified_depol[d_start:d_end+1]
        depol_times = modified_depol_times[d_start:d_end+1]
        depol_integral = np.abs(integrate.trapz(depol_data, depol_times * 1000))
        
        # Calculate capacitance
        params = analysis.params
        voltage_diff = abs(params['V2'] - params['V0'])
        capacitance = abs(hyperpol_integral - depol_integral) / voltage_diff if voltage_diff > 0 else 0
        
        result = {
            'hyperpol_integral': float(hyperpol_integral),
            'depol_integral': float(depol_integral),
            'capacitance': float(capacitance),
            'method_used': method,
            'ranges_used': ranges
        }
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Integral calculation error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Calculation failed', 'message': str(e)}), 500


@bp.route('/results/<analysis_id>', methods=['GET'])
def get_analysis_results(analysis_id):
    """
    Get analysis results by ID
    
    Returns:
    {
        "analysis_id": "uuid-string",
        "file_id": "uuid-string",
        "params": {...},
        "results": {...},
        "created_at": "timestamp"
    }
    """
    try:
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify({
            'analysis_id': str(analysis.id),
            'file_id': str(analysis.file_id),
            'params': analysis.params,
            'results': analysis.results,
            'processing_time': analysis.processing_time,
            'created_at': analysis.created_at.isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error fetching results: {str(e)}")
        return jsonify({'error': 'Failed to fetch results', 'message': str(e)}), 500


@bp.route('/list', methods=['GET'])
def list_analyses():
    """
    List all analyses (optionally filter by file_id)
    
    Query params:
    - file_id: Filter by file ID
    - limit: Max results (default 50)
    - offset: Pagination offset
    
    Returns:
    {
        "analyses": [...],
        "total": int,
        "limit": int,
        "offset": int
    }
    """
    try:
        file_id = request.args.get('file_id')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        query = AnalysisResult.query
        
        if file_id:
            query = query.filter_by(file_id=file_id)
        
        total = query.count()
        analyses = query.order_by(AnalysisResult.created_at.desc())\
                       .limit(limit)\
                       .offset(offset)\
                       .all()
        
        return jsonify({
            'analyses': [{
                'analysis_id': str(a.id),
                'file_id': str(a.file_id),
                'created_at': a.created_at.isoformat(),
                'processing_time': a.processing_time
            } for a in analyses],
            'total': total,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing analyses: {str(e)}")
        return jsonify({'error': 'Failed to list analyses', 'message': str(e)}), 500


@bp.route('/delete/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete an analysis result"""
    try:
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        db.session.delete(analysis)
        db.session.commit()
        
        return jsonify({'message': 'Analysis deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting analysis: {str(e)}")
        return jsonify({'error': 'Failed to delete analysis', 'message': str(e)}), 500

