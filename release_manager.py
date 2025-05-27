#!/usr/bin/env python3
"""
Universal Release Manager for Signal Analyzer
Supports multiple distribution platforms: GitHub, Google Drive, OneDrive, FTP, etc.
"""

import os
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime
import subprocess
import zipfile
from typing import Dict, List, Optional
import configparser

class ReleaseManager:
    def __init__(self, config_file: str = "release_config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self.project_root = Path(__file__).parent
        self.load_config()
        
    def load_config(self):
        """Load release configuration"""
        if self.config_file.exists():
            self.config.read(self.config_file)
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration file"""
        self.config['DEFAULT'] = {
            'app_name': 'SignalAnalyzer',
            'version': '1.0.0',
            'description': 'Signal Analyzer - Advanced Signal Processing Tool'
        }
        
        self.config['github'] = {
            'enabled': 'true',
            'repo': 'your-username/signal-analyzer',
            'token': 'your-github-token-here'
        }
        
        self.config['google_drive'] = {
            'enabled': 'false',
            'folder_id': 'your-google-drive-folder-id',
            'credentials_file': 'credentials.json'
        }
        
        self.config['ftp'] = {
            'enabled': 'false',
            'host': 'your-ftp-host.com',
            'username': 'your-username',
            'password': 'your-password',
            'path': '/public_html/downloads/'
        }
        
        self.config['webhook'] = {
            'enabled': 'false',
            'url': 'https://your-webhook-url.com/notify',
            'secret': 'your-webhook-secret'
        }
        
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        
        print(f"Created default config file: {self.config_file}")
        print("Please edit the configuration file with your actual settings.")
    
    def build_application(self) -> Optional[Path]:
        """Build the application and return path to distribution file"""
        print("🔨 Building application...")
        
        try:
            # Run the build script
            result = subprocess.run([
                'python', 'build.py'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                print(f"❌ Build failed: {result.stderr}")
                return None
            
            # Find the created ZIP file
            dist_dir = self.project_root / 'dist'
            zip_files = list(dist_dir.glob('*.zip'))
            
            if not zip_files:
                print("❌ No ZIP file found after build")
                return None
            
            # Return the most recent ZIP file
            latest_zip = max(zip_files, key=lambda x: x.stat().st_mtime)
            print(f"✅ Build completed: {latest_zip}")
            return latest_zip
            
        except Exception as e:
            print(f"❌ Build error: {e}")
            return None
    
    def upload_to_github(self, zip_path: Path, version: str) -> bool:
        """Upload release to GitHub"""
        if not self.config.getboolean('github', 'enabled', fallback=False):
            return True
        
        print("📤 Uploading to GitHub...")
        
        try:
            repo = self.config.get('github', 'repo')
            token = self.config.get('github', 'token')
            
            if not repo or not token or token == 'your-github-token-here':
                print("⚠️  GitHub not configured properly")
                return False
            
            # Create release
            release_data = {
                'tag_name': f'v{version}',
                'name': f'Signal Analyzer v{version}',
                'body': self.generate_release_notes(version),
                'draft': False,
                'prerelease': False
            }
            
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Create the release
            response = requests.post(
                f'https://api.github.com/repos/{repo}/releases',
                json=release_data,
                headers=headers
            )
            
            if response.status_code != 201:
                print(f"❌ Failed to create GitHub release: {response.text}")
                return False
            
            release_info = response.json()
            upload_url = release_info['upload_url'].replace('{?name,label}', '')
            
            # Upload the ZIP file
            with open(zip_path, 'rb') as f:
                upload_response = requests.post(
                    f"{upload_url}?name={zip_path.name}",
                    data=f.read(),
                    headers={
                        **headers,
                        'Content-Type': 'application/zip'
                    }
                )
            
            if upload_response.status_code == 201:
                print(f"✅ Successfully uploaded to GitHub: {release_info['html_url']}")
                return True
            else:
                print(f"❌ Failed to upload file: {upload_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ GitHub upload error: {e}")
            return False
    
    def upload_to_google_drive(self, zip_path: Path) -> bool:
        """Upload to Google Drive (requires google-api-python-client)"""
        if not self.config.getboolean('google_drive', 'enabled', fallback=False):
            return True
        
        print("📤 Uploading to Google Drive...")
        
        try:
            from googleapiclient.discovery import build
            from google.oauth2.service_account import Credentials
            from googleapiclient.http import MediaFileUpload
            
            credentials_file = self.config.get('google_drive', 'credentials_file')
            folder_id = self.config.get('google_drive', 'folder_id')
            
            if not Path(credentials_file).exists():
                print(f"⚠️  Google Drive credentials not found: {credentials_file}")
                return False
            
            # Authenticate
            credentials = Credentials.from_service_account_file(
                credentials_file,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            service = build('drive', 'v3', credentials=credentials)
            
            # Upload file
            file_metadata = {
                'name': zip_path.name,
                'parents': [folder_id] if folder_id else []
            }
            
            media = MediaFileUpload(str(zip_path), mimetype='application/zip')
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()
            
            print(f"✅ Successfully uploaded to Google Drive: {file.get('webViewLink')}")
            return True
            
        except ImportError:
            print("⚠️  Google Drive API not installed. Run: pip install google-api-python-client google-auth")
            return False
        except Exception as e:
            print(f"❌ Google Drive upload error: {e}")
            return False
    
    def upload_to_ftp(self, zip_path: Path) -> bool:
        """Upload to FTP server"""
        if not self.config.getboolean('ftp', 'enabled', fallback=False):
            return True
        
        print("📤 Uploading to FTP...")
        
        try:
            import ftplib
            
            host = self.config.get('ftp', 'host')
            username = self.config.get('ftp', 'username')
            password = self.config.get('ftp', 'password')
            remote_path = self.config.get('ftp', 'path', fallback='/')
            
            with ftplib.FTP(host) as ftp:
                ftp.login(username, password)
                ftp.cwd(remote_path)
                
                with open(zip_path, 'rb') as f:
                    ftp.storbinary(f'STOR {zip_path.name}', f)
                
                print(f"✅ Successfully uploaded to FTP: {host}{remote_path}{zip_path.name}")
                return True
                
        except Exception as e:
            print(f"❌ FTP upload error: {e}")
            return False
    
    def send_webhook_notification(self, version: str, download_urls: List[str]) -> bool:
        """Send webhook notification about new release"""
        if not self.config.getboolean('webhook', 'enabled', fallback=False):
            return True
        
        print("📨 Sending webhook notification...")
        
        try:
            webhook_url = self.config.get('webhook', 'url')
            webhook_secret = self.config.get('webhook', 'secret')
            
            payload = {
                'app_name': self.config.get('DEFAULT', 'app_name'),
                'version': version,
                'release_date': datetime.now().isoformat(),
                'download_urls': download_urls,
                'description': self.config.get('DEFAULT', 'description')
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Webhook-Secret': webhook_secret
            }
            
            response = requests.post(webhook_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                print("✅ Webhook notification sent successfully")
                return True
            else:
                print(f"⚠️  Webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Webhook error: {e}")
            return False
    
    def generate_release_notes(self, version: str) -> str:
        """Generate release notes"""
        return f"""## Signal Analyzer v{version}

### What's New
- Latest version with improved performance and stability
- Updated signal processing algorithms
- Enhanced user interface

### Download Instructions
1. Download the SignalAnalyzer ZIP file
2. Extract to a folder on your computer
3. Run SignalAnalyzer.exe

### System Requirements
- Windows 10 or later
- No additional software installation required

### For Educators
This is the latest stable version for classroom use. Students can download and run directly.

**Release Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Build:** Automated build from latest source code
"""
    
    def create_download_page(self, version: str, download_urls: List[str]) -> Path:
        """Create a simple HTML download page"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analyzer - Download</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; background: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .download-section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .download-btn {{ display: inline-block; background: #007cba; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }}
        .download-btn:hover {{ background: #005a87; }}
        .instructions {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Signal Analyzer</h1>
        <h2>Version {version}</h2>
        <p>Advanced Signal Processing Tool for Education</p>
    </div>
    
    <div class="download-section">
        <h3>📥 Download Latest Version</h3>
        <p>Current Version: <strong>{version}</strong></p>
        <p>Release Date: <strong>{datetime.now().strftime('%Y-%m-%d')}</strong></p>
        
        <div class="downloads">
            {"".join([f'<a href="{url}" class="download-btn">Download from {url.split("/")[2] if "http" in url else "Server"}</a>' for url in download_urls])}
        </div>
    </div>
    
    <div class="instructions">
        <h3>🚀 Installation Instructions</h3>
        <ol>
            <li>Click the download button above</li>
            <li>Save the ZIP file to your computer</li>
            <li>Extract/unzip the file to a folder</li>
            <li>Double-click <strong>SignalAnalyzer.exe</strong> to run</li>
        </ol>
        
        <h3>💻 System Requirements</h3>
        <ul>
            <li>Windows 10 or later</li>
            <li>No additional software installation required</li>
            <li>Recommended: 4GB RAM, 100MB free disk space</li>
        </ul>
        
        <h3>📚 For Teachers</h3>
        <p>This application is ready for classroom use. Students can download and run it directly without administrative privileges.</p>
    </div>
    
    <div class="download-section">
        <h3>🔄 Auto-Update Information</h3>
        <p>This page is automatically updated when new versions are released. Bookmark this page for easy access to the latest version.</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        
        html_file = self.project_root / 'download.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Created download page: {html_file}")
        return html_file
    
    def release(self, version: Optional[str] = None) -> bool:
        """Main release process"""
        if not version:
            version = self.config.get('DEFAULT', 'version', fallback='1.0.0')
        
        print(f"🚀 Starting release process for version {version}")
        print("=" * 60)
        
        # Step 1: Build application
        zip_path = self.build_application()
        if not zip_path:
            return False
        
        # Step 2: Upload to various platforms
        download_urls = []
        
        # GitHub
        if self.upload_to_github(zip_path, version):
            repo = self.config.get('github', 'repo', fallback='')
            if repo:
                download_urls.append(f"https://github.com/{repo}/releases/latest")
        
        # Google Drive
        if self.upload_to_google_drive(zip_path):
            download_urls.append("Google Drive")
        
        # FTP
        if self.upload_to_ftp(zip_path):
            host = self.config.get('ftp', 'host', fallback='')
            path = self.config.get('ftp', 'path', fallback='/')
            if host:
                download_urls.append(f"http://{host}{path}{zip_path.name}")
        
        # Step 3: Create download page
        if download_urls:
            self.create_download_page(version, download_urls)
        
        # Step 4: Send notifications
        self.send_webhook_notification(version, download_urls)
        
        print("=" * 60)
        print("🎉 Release process completed!")
        print(f"📦 Package: {zip_path}")
        print(f"🔗 Download URLs: {', '.join(download_urls)}")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Release Manager for Signal Analyzer')
    parser.add_argument('--version', help='Version number (e.g., 1.0.0)')
    parser.add_argument('--config', default='release_config.ini', help='Configuration file')
    parser.add_argument('--build-only', action='store_true', help='Only build, do not upload')
    
    args = parser.parse_args()
    
    manager = ReleaseManager(args.config)
    
    if args.build_only:
        zip_path = manager.build_application()
        if zip_path:
            print(f"✅ Build completed: {zip_path}")
        else:
            print("❌ Build failed")
    else:
        success = manager.release(args.version)
        if not success:
            exit(1)

if __name__ == "__main__":
    main()