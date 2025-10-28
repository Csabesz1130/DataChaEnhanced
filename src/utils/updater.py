"""
Auto-updater module for Signal Analyzer
Handles checking for updates, downloading, and applying them
"""

import requests
import json
import os
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Optional, Callable
import logging

try:
    from packaging import version
except ImportError:
    # Fallback for simple version comparison
    class version:
        @staticmethod
        def parse(v):
            return tuple(map(int, v.split('.')))

logger = logging.getLogger(__name__)


class AutoUpdater:
    """
    Automatic update checker and installer
    
    Usage:
        updater = AutoUpdater("1.0.2", "yourusername/DataChaEnhanced")
        update_info = updater.check_for_updates()
        
        if update_info['available']:
            update_file = updater.download_update(
                update_info['download_url'],
                progress_callback=my_progress_function
            )
            updater.apply_update(update_file)
    """
    
    def __init__(self, current_version: str, repo: str = "yourusername/DataChaEnhanced"):
        """
        Initialize the auto-updater
        
        Args:
            current_version: Current version string (e.g., "1.0.2")
            repo: GitHub repository in format "username/reponame"
        """
        try:
            self.current_version = version.parse(current_version)
        except:
            self.current_version = tuple(map(int, current_version.split('.')))
            
        self.repo = repo
        self.github_api = f"https://api.github.com/repos/{repo}/releases/latest"
        self.update_check_timeout = 10  # seconds
        
    def check_for_updates(self, prerelease: bool = False) -> Dict:
        """
        Check if a new version is available
        
        Args:
            prerelease: Include pre-release versions
            
        Returns:
            Dictionary with update information:
            {
                'available': bool,
                'version': str,
                'download_url': str,
                'changelog': str,
                'size': int (bytes),
                'published_at': str
            }
        """
        logger.info("Checking for updates...")
        
        try:
            # Use releases/latest for stable releases only
            api_url = self.github_api
            if prerelease:
                api_url = f"https://api.github.com/repos/{self.repo}/releases"
            
            response = requests.get(api_url, timeout=self.update_check_timeout)
            
            if response.status_code != 200:
                logger.warning(f"Update check failed with status {response.status_code}")
                return {'available': False, 'error': 'Failed to check for updates'}
            
            if prerelease:
                releases = response.json()
                if not releases:
                    return {'available': False}
                latest_release = releases[0]
            else:
                latest_release = response.json()
            
            # Parse version
            tag_name = latest_release['tag_name'].lstrip('v')
            try:
                latest_version = version.parse(tag_name)
            except:
                latest_version = tuple(map(int, tag_name.split('.')))
            
            # Compare versions
            if latest_version > self.current_version:
                download_url, file_size = self._get_download_url(latest_release)
                
                if download_url:
                    logger.info(f"Update available: {tag_name}")
                    return {
                        'available': True,
                        'version': tag_name,
                        'download_url': download_url,
                        'changelog': latest_release.get('body', 'No changelog available'),
                        'size': file_size,
                        'published_at': latest_release.get('published_at', ''),
                        'html_url': latest_release.get('html_url', '')
                    }
                else:
                    logger.warning("Update available but no compatible download found")
                    return {'available': False, 'error': 'No compatible download'}
            else:
                logger.info("Application is up to date")
                return {'available': False, 'reason': 'up_to_date'}
                
        except requests.exceptions.Timeout:
            logger.warning("Update check timed out")
            return {'available': False, 'error': 'Timeout'}
        except requests.exceptions.ConnectionError:
            logger.warning("No internet connection for update check")
            return {'available': False, 'error': 'No connection'}
        except Exception as e:
            logger.error(f"Error checking for updates: {e}", exc_info=True)
            return {'available': False, 'error': str(e)}
    
    def _get_download_url(self, release_data: Dict) -> tuple[Optional[str], int]:
        """
        Extract download URL for current platform
        
        Returns:
            Tuple of (download_url, file_size) or (None, 0)
        """
        import platform
        system = platform.system().lower()
        
        for asset in release_data.get('assets', []):
            asset_name = asset['name'].lower()
            
            # Match platform-specific downloads
            if system in asset_name and (asset_name.endswith('.zip') or 
                                        asset_name.endswith('.exe')):
                return asset['browser_download_url'], asset.get('size', 0)
        
        # Fallback: return first zip file
        for asset in release_data.get('assets', []):
            if asset['name'].endswith('.zip'):
                return asset['browser_download_url'], asset.get('size', 0)
        
        return None, 0
    
    def download_update(self, 
                       download_url: str, 
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[Path]:
        """
        Download update file with progress reporting
        
        Args:
            download_url: URL to download from
            progress_callback: Function called with (downloaded_bytes, total_bytes)
            
        Returns:
            Path to downloaded file or None if failed
        """
        temp_path = Path("temp_update.zip")
        
        try:
            logger.info(f"Downloading update from {download_url}")
            
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            try:
                                progress_callback(downloaded, total_size)
                            except Exception as e:
                                logger.warning(f"Progress callback error: {e}")
            
            # Verify download
            if total_size > 0 and downloaded != total_size:
                logger.error(f"Download incomplete: {downloaded}/{total_size} bytes")
                temp_path.unlink(missing_ok=True)
                return None
            
            logger.info(f"Download complete: {downloaded} bytes")
            return temp_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return None
    
    def verify_download(self, file_path: Path, expected_sha256: Optional[str] = None) -> bool:
        """
        Verify integrity of downloaded file
        
        Args:
            file_path: Path to file to verify
            expected_sha256: Expected SHA-256 hash (if available)
            
        Returns:
            True if verification passed
        """
        if not file_path.exists():
            return False
        
        try:
            # Calculate SHA-256 hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            calculated_hash = sha256_hash.hexdigest()
            logger.info(f"File SHA-256: {calculated_hash}")
            
            if expected_sha256:
                if calculated_hash.lower() == expected_sha256.lower():
                    logger.info("Hash verification passed")
                    return True
                else:
                    logger.error("Hash verification failed!")
                    return False
            
            # If no expected hash, just verify file is readable and not empty
            return file_path.stat().st_size > 0
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def apply_update(self, update_file: Path, backup: bool = True) -> bool:
        """
        Apply the downloaded update
        
        Creates a batch script that:
        1. Waits for app to close
        2. Backs up current version (optional)
        3. Extracts new version
        4. Restarts application
        5. Cleans up
        
        Args:
            update_file: Path to downloaded update file
            backup: Create backup of current version
            
        Returns:
            True if update process initiated successfully
        """
        import platform
        
        if not update_file.exists():
            logger.error(f"Update file not found: {update_file}")
            return False
        
        try:
            if platform.system() == "Windows":
                return self._apply_update_windows(update_file, backup)
            else:
                return self._apply_update_unix(update_file, backup)
        except Exception as e:
            logger.error(f"Failed to apply update: {e}", exc_info=True)
            return False
    
    def _apply_update_windows(self, update_file: Path, backup: bool) -> bool:
        """Apply update on Windows"""
        updater_script = Path("updater.bat")
        app_exe = "SignalAnalyzer.exe"
        
        # Create backup directory
        backup_cmd = ""
        if backup:
            backup_cmd = f"""
echo Creating backup...
if exist "backup" rmdir /s /q "backup"
mkdir "backup"
xcopy /e /i /y *.* "backup" >nul 2>&1
"""
        
        script_content = f"""@echo off
title Updating Signal Analyzer
echo ========================================
echo Signal Analyzer Update
echo ========================================
echo.

REM Wait for application to close
echo Waiting for application to close...
timeout /t 3 /nobreak > nul

REM Kill process if still running
taskkill /f /im {app_exe} 2>nul
timeout /t 1 /nobreak > nul

{backup_cmd}

REM Extract update
echo Extracting update...
powershell -Command "Expand-Archive -Path '{update_file}' -DestinationPath '.' -Force"

if errorlevel 1 (
    echo ERROR: Failed to extract update!
    if exist "backup" (
        echo Restoring from backup...
        xcopy /e /i /y "backup\\*.*" "." >nul
        rmdir /s /q "backup"
    )
    pause
    exit /b 1
)

REM Clean up
echo Cleaning up...
del /f /q "{update_file}" 2>nul

REM Start application
echo Starting Signal Analyzer...
timeout /t 2 /nobreak > nul
start "" "{app_exe}"

REM Clean up backup after successful update
if exist "backup" rmdir /s /q "backup"

REM Self-destruct
timeout /t 2 /nobreak > nul
del "%~f0"
"""
        
        try:
            updater_script.write_text(script_content, encoding='utf-8')
            logger.info(f"Created updater script: {updater_script}")
            
            # Launch updater in new console
            subprocess.Popen(
                [str(updater_script)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                close_fds=True
            )
            
            logger.info("Update process initiated. Application will restart.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create updater script: {e}")
            return False
    
    def _apply_update_unix(self, update_file: Path, backup: bool) -> bool:
        """Apply update on Unix-like systems (Linux, macOS)"""
        updater_script = Path("updater.sh")
        app_binary = "SignalAnalyzer"
        
        backup_cmd = ""
        if backup:
            backup_cmd = """
echo "Creating backup..."
rm -rf backup
mkdir -p backup
cp -r ./* backup/ 2>/dev/null || true
"""
        
        script_content = f"""#!/bin/bash

echo "========================================"
echo "Signal Analyzer Update"
echo "========================================"
echo

# Wait for application to close
echo "Waiting for application to close..."
sleep 3

# Kill process if still running
killall {app_binary} 2>/dev/null || true
sleep 1

{backup_cmd}

# Extract update
echo "Extracting update..."
unzip -o "{update_file}" -d .

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract update!"
    if [ -d "backup" ]; then
        echo "Restoring from backup..."
        cp -r backup/* .
        rm -rf backup
    fi
    exit 1
fi

# Set permissions
chmod +x {app_binary}

# Clean up
echo "Cleaning up..."
rm -f "{update_file}"

# Start application
echo "Starting Signal Analyzer..."
sleep 2
./{app_binary} &

# Clean up backup after successful update
rm -rf backup

# Self-destruct
rm -- "$0"
"""
        
        try:
            updater_script.write_text(script_content, encoding='utf-8')
            updater_script.chmod(0o755)
            logger.info(f"Created updater script: {updater_script}")
            
            # Launch updater
            subprocess.Popen(
                [str(updater_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            logger.info("Update process initiated. Application will restart.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create updater script: {e}")
            return False


def check_for_updates_silent(current_version: str, repo: str) -> Optional[Dict]:
    """
    Convenience function for silent update checking
    
    Returns update info if available, None otherwise
    """
    try:
        updater = AutoUpdater(current_version, repo)
        update_info = updater.check_for_updates()
        
        if update_info.get('available'):
            return update_info
        return None
    except:
        return None


if __name__ == "__main__":
    # Test the updater
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_version = "1.0.0"  # Test with older version
    test_repo = "yourusername/DataChaEnhanced"
    
    print(f"Testing updater with version {test_version}")
    print(f"Repository: {test_repo}")
    print()
    
    updater = AutoUpdater(test_version, test_repo)
    
    print("Checking for updates...")
    update_info = updater.check_for_updates()
    
    if update_info.get('available'):
        print(f"\n✅ Update available!")
        print(f"   Current version: {test_version}")
        print(f"   Latest version: {update_info['version']}")
        print(f"   Download size: {update_info.get('size', 0) / 1024 / 1024:.1f} MB")
        print(f"   Changelog:\n{update_info['changelog'][:200]}...")
        
        # Test download (commented out to avoid actual download)
        # print("\nDownload test (skipped)")
        # update_file = updater.download_update(update_info['download_url'])
        
    else:
        print(f"\n✅ No updates available")
        if 'error' in update_info:
            print(f"   Error: {update_info['error']}")
        elif 'reason' in update_info:
            print(f"   Reason: {update_info['reason']}")

