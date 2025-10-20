"""
Export Backup Manager for DataChaEnhanced
=========================================
Location: src/excel_export/export_backup_manager.py

This module handles creating and restoring backups of analysis data before exports.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ExportBackupManager:
    """Manages backup and restore functionality for export operations."""
    
    def __init__(self, backup_dir: str = "excel_export_backups"):
        """
        Initialize the backup manager.
        
        Args:
            backup_dir: Directory to store backup files
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backup_age_days = 30
        
    def create_backup(self, analysis_data: Dict[str, Any], filename: str) -> Optional[str]:
        """
        Create a backup of current analysis state.
        
        Args:
            analysis_data: Dictionary containing current analysis state
            filename: Original filename for reference
            
        Returns:
            str: Path to backup file if successful, None otherwise
        """
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{timestamp}_{Path(filename).stem}.json"
            backup_path = self.backup_dir / backup_filename
            
            # Prepare backup data
            backup_data = {
                'timestamp': timestamp,
                'original_filename': filename,
                'backup_created': datetime.now().isoformat(),
                'analysis_data': analysis_data
            }
            
            # Write backup file
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backup created: {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return None
    
    def restore_backup(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        Restore analysis state from backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dict: Restored analysis data if successful, None otherwise
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return None
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            logger.info(f"Backup restored from: {backup_path}")
            return backup_data.get('analysis_data', {})
            
        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            return None
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backup files with metadata.
        
        Returns:
            List of dictionaries containing backup information
        """
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("backup_*.json"):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_data = json.load(f)
                    
                    file_stat = backup_file.stat()
                    backups.append({
                        'path': str(backup_file),
                        'filename': backup_file.name,
                        'original_filename': backup_data.get('original_filename', 'Unknown'),
                        'created': backup_data.get('backup_created', 'Unknown'),
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Error reading backup file {backup_file}: {str(e)}")
                    continue
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {str(e)}")
            return []
    
    def delete_backup(self, backup_path: str) -> bool:
        """
        Delete a specific backup file.
        
        Args:
            backup_path: Path to backup file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            backup_file = Path(backup_path)
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"Backup deleted: {backup_path}")
                return True
            else:
                logger.warning(f"Backup file not found: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting backup: {str(e)}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Remove backup files older than max_backup_age_days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_backup_age_days)
            deleted_count = 0
            
            for backup_file in self.backup_dir.glob("backup_*.json"):
                try:
                    file_modified = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_modified < cutoff_date:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    logger.warning(f"Error checking backup file {backup_file}: {str(e)}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")
                
        except Exception as e:
            logger.error(f"Error during backup cleanup: {str(e)}")
    
    def get_backup_info(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dict: Backup information if successful, None otherwise
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return None
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            file_stat = backup_file.stat()
            return {
                'path': str(backup_file),
                'filename': backup_file.name,
                'original_filename': backup_data.get('original_filename', 'Unknown'),
                'created': backup_data.get('backup_created', 'Unknown'),
                'size': file_stat.st_size,
                'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'has_hyperpol_data': 'hyperpol' in backup_data.get('analysis_data', {}),
                'has_depol_data': 'depol' in backup_data.get('analysis_data', {}),
                'has_fitting_data': any('linear' in curve_data or 'exponential' in curve_data 
                                     for curve_data in backup_data.get('analysis_data', {}).values() 
                                     if isinstance(curve_data, dict))
            }
            
        except Exception as e:
            logger.error(f"Error getting backup info: {str(e)}")
            return None

# Global backup manager instance
backup_manager = ExportBackupManager()
