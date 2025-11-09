"""
File upload/download API endpoints
"""

from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import uuid
import os
from pathlib import Path

from backend.utils.file_handler import allowed_file, save_uploaded_file, get_file_path
from backend.utils.db import db, UploadedFile
from src.io_utils.io_utils import ATFHandler

bp = Blueprint('files', __name__)


@bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload an ATF file
    
    Form data:
    - file: The ATF file (multipart/form-data)
    
    Returns:
    {
        "file_id": "uuid-string",
        "filename": "original_filename.atf",
        "file_size": 12345,
        "data_info": {
            "num_points": 1000,
            "duration": 5.0,
            "sampling_rate": 200
        }
    }
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: .atf, .txt, .csv'}), 400
        
        # Save file
        file_id, file_path, file_size = save_uploaded_file(file)
        
        # Parse file to get data info
        try:
            handler = ATFHandler(file_path)
            handler.load_atf()
            
            if handler.data is not None and len(handler.data.shape) > 1:
                # First column is typically time
                time_data = handler.get_column('time') if 'time' in [h.lower() for h in handler.headers] else handler.data[:, 0]
                # Get signal data (second column or first trace)
                signal_data = handler.get_column('#1') if len(handler.signal_map) > 0 else handler.data[:, 1] if handler.data.shape[1] > 1 else handler.data[:, 0]
                
                num_points = len(signal_data)
                duration = float(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0
                sampling_rate = num_points / duration if duration > 0 else 0
                
                data_info = {
                    'num_points': num_points,
                    'duration': duration,
                    'sampling_rate': sampling_rate
                }
            else:
                data_info = None
        except Exception as e:
            current_app.logger.warning(f"Could not parse file info: {str(e)}")
            data_info = None
        
        # Save to database
        uploaded_file = UploadedFile(
            id=file_id,
            filename=secure_filename(file.filename),
            file_path=file_path,
            file_size=file_size,
            data_info=data_info
        )
        db.session.add(uploaded_file)
        db.session.commit()
        
        current_app.logger.info(f"File uploaded: {file_id} - {file.filename}")
        
        return jsonify({
            'file_id': str(file_id),
            'filename': file.filename,
            'file_size': file_size,
            'data_info': data_info
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Upload failed', 'message': str(e)}), 500


@bp.route('/<file_id>', methods=['GET'])
def get_file_info(file_id):
    """
    Get file information
    
    Returns:
    {
        "file_id": "uuid",
        "filename": "file.atf",
        "file_size": 12345,
        "uploaded_at": "timestamp",
        "data_info": {...}
    }
    """
    try:
        file_record = UploadedFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        
        return jsonify({
            'file_id': str(file_record.id),
            'filename': file_record.filename,
            'file_size': file_record.file_size,
            'uploaded_at': file_record.uploaded_at.isoformat(),
            'data_info': file_record.data_info
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting file info: {str(e)}")
        return jsonify({'error': 'Failed to get file info'}), 500


@bp.route('/<file_id>/data', methods=['GET'])
def get_file_data_endpoint(file_id):
    """
    Get parsed file data (signal and time arrays)
    
    Returns:
    {
        "data": [array],
        "time_data": [array],
        "metadata": {...}
    }
    """
    try:
        file_record = UploadedFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        
        # Parse file
        handler = ATFHandler(file_record.file_path)
        handler.load_atf()
        
        if handler.data is not None and len(handler.data.shape) > 1:
            # First column is typically time
            time_data = handler.get_column('time') if 'time' in [h.lower() for h in handler.headers] else handler.data[:, 0]
            # Get signal data (second column or first trace)
            signal_data = handler.get_column('#1') if len(handler.signal_map) > 0 else handler.data[:, 1] if handler.data.shape[1] > 1 else handler.data[:, 0]
            
            return jsonify({
                'data': signal_data.tolist(),
                'time_data': time_data.tolist(),
                'metadata': file_record.data_info
            })
        else:
            return jsonify({'error': 'No data found in file'}), 400
        
    except Exception as e:
        current_app.logger.error(f"Error reading file data: {str(e)}")
        return jsonify({'error': 'Failed to read file data', 'message': str(e)}), 500


@bp.route('/<file_id>/download', methods=['GET'])
def download_file(file_id):
    """Download the original uploaded file"""
    try:
        file_record = UploadedFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        
        if not os.path.exists(file_record.file_path):
            return jsonify({'error': 'File no longer exists'}), 404
        
        return send_file(
            file_record.file_path,
            as_attachment=True,
            download_name=file_record.filename
        )
        
    except Exception as e:
        current_app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500


@bp.route('/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete uploaded file"""
    try:
        file_record = UploadedFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        
        # Delete physical file
        if os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
        
        # Delete from database
        db.session.delete(file_record)
        db.session.commit()
        
        current_app.logger.info(f"File deleted: {file_id}")
        
        return jsonify({'message': 'File deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': 'Delete failed'}), 500


@bp.route('/list', methods=['GET'])
def list_files():
    """
    List all uploaded files
    
    Query params:
    - limit: Max results (default 50)
    - offset: Pagination offset
    
    Returns:
    {
        "files": [...],
        "total": int,
        "limit": int,
        "offset": int
    }
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        total = UploadedFile.query.count()
        files = UploadedFile.query.order_by(UploadedFile.uploaded_at.desc())\
                                   .limit(limit)\
                                   .offset(offset)\
                                   .all()
        
        return jsonify({
            'files': [{
                'file_id': str(f.id),
                'filename': f.filename,
                'file_size': f.file_size,
                'uploaded_at': f.uploaded_at.isoformat(),
                'data_info': f.data_info
            } for f in files],
            'total': total,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing files: {str(e)}")
        return jsonify({'error': 'Failed to list files'}), 500

