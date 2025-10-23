"""
Export Backup Manager
Provides backup functionality for Excel exports
"""

import os
import json
import shutil
from datetime import datetime
from src.utils.logger import app_logger

class ExportBackupManager:
    """Manages backups of export data."""
    
    def __init__(self, backup_dir=None):
        """Initialize the backup manager.
        
        Args:
            backup_dir: Directory for storing backups. If None, uses default.
        """
        if backup_dir is None:
            # Use workspace/backups as default
            backup_dir = os.path.join(os.getcwd(), "backups", "excel_exports")
        
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, export_data, filename):
        """Create a backup of export data before exporting.
        
        Args:
            export_data: Dictionary containing export data
            filename: Original filename
            
        Returns:
            str: Path to backup file, or None if backup failed
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(filename))[0]
            backup_filename = f"{base_name}_backup_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_serializable(export_data)
            
            # Save to JSON
            with open(backup_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            app_logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            app_logger.error(f"Failed to create backup: {str(e)}")
            return None
    
    def _make_serializable(self, data):
        """Convert data to JSON-serializable format.
        
        Args:
            data: Data to convert
            
        Returns:
            Serializable version of data
        """
        import numpy as np
        
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):
            return float(data)
        else:
            return data
    
    def list_backups(self):
        """List all available backups.
        
        Returns:
            list: List of backup filenames
        """
        try:
            backups = [f for f in os.listdir(self.backup_dir) if f.endswith('.json')]
            return sorted(backups, reverse=True)
        except Exception as e:
            app_logger.error(f"Failed to list backups: {str(e)}")
            return []

# Global backup manager instance
backup_manager = ExportBackupManager()
