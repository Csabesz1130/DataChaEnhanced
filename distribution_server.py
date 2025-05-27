#!/usr/bin/env python3
"""
Simple Distribution Web Server for Signal Analyzer
Provides automatic download page and file serving
"""

import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, send_file, request, jsonify
import argparse

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('releases')
UPLOAD_FOLDER.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analyzer - Download Center</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 10px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .download-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .download-btn {
            display: inline-block;
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 25px;
            margin: 10px;
            font-weight: bold;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        }
        .version-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .info-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }
        .info-card h4 {
            margin: 0 0 10px 0;
            color: #667eea;
        }
        .info-card p {
            margin: 0;
            font-size: 1.1em;
            font-weight: bold;
        }
        .instructions {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .instructions h3 {
            color: #1976d2;
            margin-top: 0;
        }
        .instructions ol {
            padding-left: 20px;
        }
        .instructions li {
            margin: 8px 0;
            line-height: 1.4;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background: #28a745;
            box-shadow: 0 0 5px #28a745;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
        }
        .auto-refresh {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        @media (max-width: 600px) {
            body { padding: 10px; }
            .container { padding: 20px; }
            .header h1 { font-size: 2em; }
            .download-btn { display: block; margin: 10px 0; text-align: center; }
        }
    </style>
    <script>
        // Auto-refresh every 5 minutes to check for updates
        setTimeout(function() {
            location.reload();
        }, 300000);
        
        // Add download tracking
        function trackDownload(filename) {
            fetch('/api/download-stats', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename, timestamp: new Date().toISOString()})
            });
        }
    </script>
</head>
<body>
    <div class="auto-refresh">
        <span class="status-indicator status-online"></span>
        Auto-refresh: ON
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🔬 Signal Analyzer</h1>
            <p>Advanced Signal Processing Tool for Education</p>
        </div>
        
        <div class="download-section">
            <h2>📥 Download Latest Version</h2>
            
            <div class="version-info">
                <div class="info-card">
                    <h4>Current Version</h4>
                    <p>{{ latest_version }}</p>
                </div>
                <div class="info-card">
                    <h4>Release Date</h4>
                    <p>{{ release_date }}</p>
                </div>
                <div class="info-card">
                    <h4>File Size</h4>
                    <p>{{ file_size }}</p>
                </div>
                <div class="info-card">
                    <h4>Downloads</h4>
                    <p>{{ download_count }}</p>
                </div>
            </div>
            
            <div style="text-align: center; margin: 25px 0;">
                {% for file in download_files %}
                <a href="/download/{{ file.name }}" 
                   class="download-btn" 
                   onclick="trackDownload('{{ file.name }}')">
                    💾 Download {{ file.name }}
                </a>
                {% endfor %}
            </div>
        </div>
        
        <div class="instructions">
            <h3>🚀 Installation Instructions</h3>
            <ol>
                <li><strong>Download:</strong> Click the download button above to get the latest version</li>
                <li><strong>Extract:</strong> Unzip the downloaded file to a folder on your computer</li>
                <li><strong>Run:</strong> Double-click <code>SignalAnalyzer.exe</code> to start the application</li>
                <li><strong>No Installation Required:</strong> The application runs directly without installation</li>
            </ol>
            
            <h3>💻 System Requirements</h3>
            <ul>
                <li>Windows 10 or later (64-bit recommended)</li>
                <li>4GB RAM minimum, 8GB recommended</li>
                <li>100MB free disk space</li>
                <li>No additional software installation required</li>
            </ul>
        </div>
        
        <div class="download-section">
            <h3>📚 For Educators</h3>
            <p>This application is designed for classroom use:</p>
            <ul>
                <li>✅ No installation required - runs directly from download</li>
                <li>✅ No administrator privileges needed</li>
                <li>✅ Portable - can be run from USB drives</li>
                <li>✅ Same version for all students</li>
                <li>✅ Automatic updates available through this page</li>
            </ul>
        </div>
        
        <div class="download-section">
            <h3>🔄 Version History</h3>
            <div style="max-height: 200px; overflow-y: auto;">
                {% for version in version_history %}
                <div style="padding: 10px; border-bottom: 1px solid #eee;">
                    <strong>{{ version.version }}</strong> - {{ version.date }}
                    <br><small>{{ version.description }}</small>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="footer">
            <p>
                🔗 This page updates automatically when new versions are released<br>
                📧 For support, contact your instructor<br>
                🕒 Last updated: {{ current_time }}
            </p>
        </div>
    </div>
</body>
</html>
"""

class DistributionServer:
    def __init__(self, upload_folder="releases"):
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(exist_ok=True)
        self.stats_file = self.upload_folder / "download_stats.json"
        self.version_file = self.upload_folder / "versions.json"
        
        # Initialize stats file
        if not self.stats_file.exists():
            self.save_stats({})
        
        # Initialize version history
        if not self.version_file.exists():
            self.save_versions([])
    
    def save_stats(self, stats):
        """Save download statistics"""
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_stats(self):
        """Load download statistics"""
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_versions(self, versions):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
    
    def load_versions(self):
        """Load version history"""
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def get_latest_files(self):
        """Get list of available download files"""
        files = []
        for file_path in self.upload_folder.glob("*.zip"):
            stats = file_path.stat()
            files.append({
                'name': file_path.name,
                'path': file_path,
                'size': stats.st_size,
                'modified': datetime.fromtimestamp(stats.st_mtime)
            })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    def extract_version_from_filename(self, filename):
        """Extract version number from filename"""
        import re
        match = re.search(r'v?(\d+\.\d+\.\d+)', filename)
        return match.group(1) if match else "Unknown"
    
    def record_download(self, filename):
        """Record a download event"""
        stats = self.load_stats()
        if filename not in stats:
            stats[filename] = 0
        stats[filename] += 1
        self.save_stats(stats)
    
    def get_download_count(self, filename=None):
        """Get download count for a file or total"""
        stats = self.load_stats()
        if filename:
            return stats.get(filename, 0)
        return sum(stats.values())

# Initialize server
server = DistributionServer()

@app.route('/')
def index():
    """Main download page"""
    files = server.get_latest_files()
    
    if not files:
        return """
        <div style="text-align: center; margin-top: 100px; font-family: Arial;">
            <h1>No releases available</h1>
            <p>Upload a release file to the 'releases' folder to get started.</p>
        </div>
        """
    
    latest_file = files[0]
    version = server.extract_version_from_filename(latest_file['name'])
    
    template_data = {
        'latest_version': version,
        'release_date': latest_file['modified'].strftime('%Y-%m-%d'),
        'file_size': server.format_file_size(latest_file['size']),
        'download_count': server.get_download_count(),
        'download_files': files[:3],  # Show top 3 files
        'version_history': server.load_versions()[-10:],  # Last 10 versions
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template_string(HTML_TEMPLATE, **template_data)

@app.route('/download/<filename>')
def download_file(filename):
    """Download a specific file"""
    file_path = server.upload_folder / filename
    
    if not file_path.exists():
        return "File not found", 404
    
    # Record the download
    server.record_download(filename)
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/download-stats', methods=['POST'])
def record_download_api():
    """API endpoint to record download statistics"""
    data = request.get_json()
    if data and 'filename' in data:
        server.record_download(data['filename'])
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400

@app.route('/api/stats')
def get_stats():
    """API endpoint to get download statistics"""
    return jsonify({
        'files': server.load_stats(),
        'total_downloads': server.get_download_count(),
        'available_files': len(server.get_latest_files())
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a new release file"""
    if 'file' not in request.files:
        return "No file provided", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    if file and file.filename.endswith('.zip'):
        # Save the file
        file_path = server.upload_folder / file.filename
        file.save(file_path)
        
        # Update version history
        versions = server.load_versions()
        version = server.extract_version_from_filename(file.filename)
        versions.append({
            'version': version,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'filename': file.filename,
            'description': f'Release {version}'
        })
        server.save_versions(versions)
        
        return f"File {file.filename} uploaded successfully", 200
    
    return "Invalid file type. Only ZIP files are allowed.", 400

def main():
    parser = argparse.ArgumentParser(description='Distribution Server for Signal Analyzer')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--upload-folder', default='releases', help='Folder for release files')
    
    args = parser.parse_args()
    
    # Update server configuration
    global server
    server = DistributionServer(args.upload_folder)
    
    print(f"🚀 Starting Distribution Server...")
    print(f"📁 Upload folder: {args.upload_folder}")
    print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
    print(f"📤 Upload files to: http://{args.host}:{args.port}/upload")
    print(f"📊 Statistics at: http://{args.host}:{args.port}/api/stats")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()