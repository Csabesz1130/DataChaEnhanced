"""
Flask Backend for Signal Analyzer Web Application
Reuses desktop analysis code from src/
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from src/
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_compress import Compress
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime

# Import configuration
from backend.config import Config

# Import routes
from backend.routes import analysis, filtering, files, export_routes
from backend.utils.db import init_db

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for frontend
CORS(app, resources={
    r"/api/*": {
        "origins": app.config['FRONTEND_URL'],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Enable compression
Compress(app)

# Initialize database
db = init_db(app)

# Register blueprints
app.register_blueprint(analysis.bp, url_prefix='/api/analysis')
app.register_blueprint(filtering.bp, url_prefix='/api/filter')
app.register_blueprint(files.bp, url_prefix='/api/files')
app.register_blueprint(export_routes.bp, url_prefix='/api/export')


# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Heroku"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


# Root endpoint
@app.route('/')
def index():
    """Root endpoint - API info"""
    return jsonify({
        'name': 'Signal Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'analysis': '/api/analysis/*',
            'filtering': '/api/filter/*',
            'files': '/api/files/*',
            'export': '/api/export/*'
        },
        'documentation': '/api/docs'
    })


# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# Request logging
@app.before_request
def log_request():
    """Log all requests for debugging"""
    if app.config['DEBUG']:
        app.logger.info(f"{request.method} {request.path}")


@app.after_request
def after_request(response):
    """Add headers and log response"""
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    
    if app.config['DEBUG']:
        app.logger.info(f"Response: {response.status_code}")
    
    return response


# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])

