"""
History API endpoints for analysis runs
"""

from flask import Blueprint, request, jsonify, current_app
from backend.utils.db import db, AnalysisRun, UploadedFile, AnalysisResult

bp = Blueprint('history', __name__)


@bp.route('/list', methods=['GET'])
def list_runs():
    """
    List all analysis runs with pagination
    
    Query params:
    - limit: Max results (default 50)
    - offset: Pagination offset (default 0)
    - file_id: Filter by file ID (optional)
    
    Returns:
    {
        "runs": [...],
        "total": int,
        "limit": int,
        "offset": int
    }
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        file_id = request.args.get('file_id')
        
        query = AnalysisRun.query
        
        if file_id:
            query = query.filter_by(file_id=file_id)
        
        total = query.count()
        runs = query.order_by(AnalysisRun.created_at.desc())\
                   .limit(limit)\
                   .offset(offset)\
                   .all()
        
        return jsonify({
            'runs': [run.to_dict() for run in runs],
            'total': total,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing runs: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to list runs', 'message': str(e)}), 500


@bp.route('/<run_id>', methods=['GET'])
def get_run(run_id):
    """
    Get a specific analysis run by ID
    First checks TinyDB cache, then falls back to database
    
    Returns:
    {
        "id": "uuid",
        "file_id": "uuid",
        "analysis_id": "uuid",
        "file_name": "filename.atf",
        "params": {...},
        "results": {...},
        "processing_time": float,
        "created_at": "timestamp"
    }
    """
    try:
        # Try TinyDB cache first
        from backend.utils.local_cache import get_cached_run
        cached_run = get_cached_run(run_id)
        if cached_run:
            current_app.logger.debug(f"Run {run_id} found in cache")
            return jsonify(cached_run)
        
        # Fall back to database
        run = AnalysisRun.query.get(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        
        run_dict = run.to_dict()
        
        # Cache it for next time
        from backend.utils.local_cache import cache_run
        cache_run(run_id, run_dict)
        
        return jsonify(run_dict)
        
    except Exception as e:
        current_app.logger.error(f"Error getting run: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get run', 'message': str(e)}), 500


@bp.route('/<run_id>', methods=['DELETE'])
def delete_run(run_id):
    """Delete an analysis run"""
    try:
        run = AnalysisRun.query.get(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        
        db.session.delete(run)
        db.session.commit()
        
        current_app.logger.info(f"Run deleted: {run_id}")
        return jsonify({'message': 'Run deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting run: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to delete run', 'message': str(e)}), 500


