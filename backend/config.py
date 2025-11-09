"""
Configuration for Flask backend
"""

import os
from datetime import timedelta


class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://localhost/signal_analyzer'
    
    # Handle Heroku's postgres:// → postgresql://
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 50 * 1024 * 1024))  # 50 MB
    ALLOWED_EXTENSIONS = {'atf', 'txt', 'csv'}
    
    # Frontend URL (for CORS)
    FRONTEND_URL = os.environ.get('FRONTEND_URL') or 'http://localhost:3000,http://localhost:3001'
    
    # Session
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Cache (Redis)
    CACHE_TYPE = os.environ.get('CACHE_TYPE') or 'simple'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    
    # AWS S3 (for production file storage)
    S3_BUCKET = os.environ.get('S3_BUCKET')
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    # Analysis settings
    DEFAULT_ANALYSIS_PARAMS = {
        'n_cycles': 2,
        't0': 20,
        't1': 100,
        't2': 100,
        't3': 1000,
        'V0': -80,
        'V1': -100,
        'V2': -20,
        'cell_area_cm2': 1e-4
    }
    
    # Celery (for async tasks)
    CELERY_BROKER_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

