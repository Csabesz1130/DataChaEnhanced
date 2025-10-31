"""
File upload and handling utilities
"""

import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app
import numpy as np

from src.io_utils.io_utils import ATFHandler


ALLOWED_EXTENSIONS = {'atf', 'txt', 'csv'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    """
    Save uploaded file with secure filename and UUID
    
    Args:
        file: Werkzeug FileStorage object
    
    Returns:
        tuple: (file_id, file_path, file_size)
    """
    if not allowed_file(file.filename):
        raise ValueError(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Generate unique ID
    file_id = str(uuid.uuid4())
    
    # Secure the filename
    original_filename = secure_filename(file.filename)
    
    # Create filename with UUID
    file_ext = original_filename.rsplit('.', 1)[1].lower()
    new_filename = f"{file_id}.{file_ext}"
    
    # Get upload folder
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save file
    file_path = os.path.join(upload_folder, new_filename)
    file.save(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Validate file size
    max_size = current_app.config['MAX_CONTENT_LENGTH']
    if file_size > max_size:
        os.remove(file_path)
        raise ValueError(f"File too large. Max size: {max_size // (1024*1024)} MB")
    
    current_app.logger.info(f"File saved: {file_id} - {original_filename} ({file_size} bytes)")
    
    return file_id, file_path, file_size


def get_file_path(file_id):
    """
    Get file path from file ID
    
    Args:
        file_id: UUID string
    
    Returns:
        str: Full file path
    """
    from backend.utils.db import UploadedFile
    
    file_record = UploadedFile.query.get(file_id)
    if not file_record:
        raise ValueError(f"File not found: {file_id}")
    
    if not os.path.exists(file_record.file_path):
        raise ValueError(f"File no longer exists: {file_record.file_path}")
    
    return file_record.file_path


def get_file_data(file_id):
    """
    Load and parse file data
    
    Args:
        file_id: UUID string
    
    Returns:
        tuple: (data, time_data, file_info)
    """
    from backend.utils.db import UploadedFile
    
    file_record = UploadedFile.query.get(file_id)
    if not file_record:
        raise ValueError(f"File not found: {file_id}")
    
    file_path = file_record.file_path
    if not os.path.exists(file_path):
        raise ValueError(f"File no longer exists: {file_path}")
    
    # Parse ATF file using desktop code
    reader = ATFHandler(file_path)
    reader.load_atf()
    data = reader.data
    time_data = data[:, 0] if data is not None and len(data.shape) > 1 else None
    
    file_info = {
        'file_id': str(file_record.id),
        'filename': file_record.filename,
        'file_size': file_record.file_size,
        'data_info': file_record.data_info
    }
    
    return data, time_data, file_info


def delete_file(file_id):
    """
    Delete file from filesystem and database
    
    Args:
        file_id: UUID string
    """
    from backend.utils.db import db, UploadedFile
    
    file_record = UploadedFile.query.get(file_id)
    if not file_record:
        raise ValueError(f"File not found: {file_id}")
    
    # Delete physical file
    if os.path.exists(file_record.file_path):
        os.remove(file_record.file_path)
        current_app.logger.info(f"Deleted file: {file_record.file_path}")
    
    # Delete from database
    db.session.delete(file_record)
    db.session.commit()
    
    current_app.logger.info(f"Deleted file record: {file_id}")


def cleanup_old_files(days=7):
    """
    Clean up files older than specified days
    
    Args:
        days: Number of days to keep files
    """
    from datetime import datetime, timedelta
    from backend.utils.db import db, UploadedFile
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    old_files = UploadedFile.query.filter(UploadedFile.uploaded_at < cutoff_date).all()
    
    count = 0
    for file_record in old_files:
        try:
            delete_file(str(file_record.id))
            count += 1
        except Exception as e:
            current_app.logger.error(f"Error deleting old file {file_record.id}: {e}")
    
    current_app.logger.info(f"Cleaned up {count} old files")
    return count

