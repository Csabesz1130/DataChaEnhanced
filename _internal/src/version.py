# src/version.py
import os
import json
import sys
import tempfile
from pathlib import Path
from src.utils.logger import app_logger

APP_VERSION = "1.0.0"  # Base version
APP_NAME = "SignalAnalyzer"
GITHUB_REPO = "Csabesz1130/DataChaEnhanced"
UPDATE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def get_version_file_path():
    """Get the path to the version file"""
    return Path(__file__).parent.parent / "version_data.json"

def get_current_version():
    """Get the actual current version, including any auto-updates"""
    version_file = get_version_file_path()
    
    # If version file exists, read from it
    if version_file.exists():
        try:
            with open(version_file, 'r') as f:
                data = json.load(f)
                return data.get('version', APP_VERSION)
        except Exception as e:
            app_logger.error(f"Error reading version file: {e}")
    
    # Otherwise return the hardcoded version
    return APP_VERSION

def save_version_info(version, update_date=None):
    """Save version information to disk"""
    version_file = get_version_file_path()
    
    data = {
        'version': version,
        'update_date': update_date
    }
    
    try:
        with open(version_file, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        app_logger.error(f"Error saving version info: {e}")
        return False