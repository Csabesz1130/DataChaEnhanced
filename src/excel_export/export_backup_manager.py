"""
Export Backup Manager for DataChaEnhanced
========================================
Location: src/excel_export/export_backup_manager.py

This module provides backup functionality for Excel export operations,
ensuring data safety before writing to Excel files.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from src.utils.logger import app_logger


class ExportBackupManager:
    """Manages backup operations for Excel export data."""
    
    def __init__(self, backup_dir: str = "export_backups"):
        """Initialize the backup manager.
        
        Args:
            backup_dir: Directory to store backup files
        """
        self.backup_dir = backup_dir
        self._ensure_backup_dir()
    
    def _ensure_backup_dir(self):
        """Ensure the backup directory exists."""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            app_logger.debug(f"Backup directory ensured: {self.backup_dir}")
        except Exception as e:
            app_logger.error(f"Failed to create backup directory: {str(e)}")
            raise
    
    def create_backup(self, export_data: Dict[str, Any], filename: str) -> Optional[str]:
        """Create a backup of export data before writing to Excel.
        
        Args:
            export_data: Dictionary containing all export data
            filename: Original filename for the export
            
        Returns:
            str: Path to the backup file if successful, None otherwise
        """
        try:
            # Generate timestamp for backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create backup filename
            base_name = os.path.splitext(filename)[0]
            backup_filename = f"{base_name}_backup_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Add metadata to export data
            backup_data = {
                'metadata': {
                    'original_filename': filename,
                    'backup_timestamp': timestamp,
                    'backup_created': datetime.now().isoformat(),
                    'export_type': export_data.get('export_type', 'unknown')
                },
                'export_data': export_data
            }
            
            # Write backup to JSON file
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            
            app_logger.info(f"Backup created successfully: {backup_path}")
            return backup_path
            
        except Exception as e:
            app_logger.error(f"Failed to create backup: {str(e)}")
            return None
    
    def list_backups(self) -> list:
        """List all available backup files.
        
        Returns:
            list: List of backup file paths
        """
        try:
            if not os.path.exists(self.backup_dir):
                return []
            
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('_backup_') and filename.endswith('.json'):
                    backup_files.append(os.path.join(self.backup_dir, filename))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return backup_files
            
        except Exception as e:
            app_logger.error(f"Failed to list backups: {str(e)}")
            return []
    
    def restore_backup(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """Restore export data from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Dict: Restored export data if successful, None otherwise
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            app_logger.info(f"Backup restored from: {backup_path}")
            return backup_data.get('export_data', {})
            
        except Exception as e:
            app_logger.error(f"Failed to restore backup: {str(e)}")
            return None
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Clean up old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent backups to keep
        """
        try:
            backup_files = self.list_backups()
            
            if len(backup_files) <= keep_count:
                return
            
            # Remove old backup files
            files_to_remove = backup_files[keep_count:]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    app_logger.debug(f"Removed old backup: {file_path}")
                except Exception as e:
                    app_logger.warning(f"Failed to remove old backup {file_path}: {str(e)}")
            
            app_logger.info(f"Cleaned up {len(files_to_remove)} old backup files")
            
        except Exception as e:
            app_logger.error(f"Failed to cleanup old backups: {str(e)}")


# Global backup manager instance
backup_manager = ExportBackupManager()
