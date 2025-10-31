"""
Database models and initialization
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()


def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    return db


# Database Models

class UploadedFile(db.Model):
    """Uploaded ATF file record"""
    __tablename__ = 'uploaded_files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer)
    data_info = db.Column(db.JSON)  # Store metadata about the file
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('AnalysisResult', backref='file', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<UploadedFile {self.filename}>'


class AnalysisResult(db.Model):
    """Analysis result record"""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id = db.Column(db.String(36), db.ForeignKey('uploaded_files.id'), nullable=False)
    params = db.Column(db.JSON, nullable=False)  # Analysis parameters
    results = db.Column(db.JSON, nullable=False)  # Analysis results (curves, integrals, etc.)
    processing_time = db.Column(db.Float)  # Processing time in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    exports = db.relationship('ExportFile', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<AnalysisResult {self.id}>'


class ExportFile(db.Model):
    """Export file record"""
    __tablename__ = 'export_files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey('analysis_results.id'), nullable=False)
    export_type = db.Column(db.String(50))  # 'excel', 'csv', etc.
    file_path = db.Column(db.String(512))
    filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ExportFile {self.filename}>'


class Session(db.Model):
    """User session record (optional - for future use)"""
    __tablename__ = 'sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_data = db.Column(db.JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    def __repr__(self):
        return f'<Session {self.id}>'

