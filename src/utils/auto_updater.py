# src/utils/auto_updater.py
import os
import sys
import tempfile
import zipfile
import subprocess
import threading
import requests
import time
import shutil
from pathlib import Path
from packaging import version
from tkinter import messagebox
from src.utils.logger import app_logger
from src.version import get_current_version, save_version_info, UPDATE_URL, GITHUB_REPO

class AutoUpdater:
    def __init__(self, root, silent=False):
        self.root = root
        self.silent = silent
        self.current_version = get_current_version()
        self.update_in_progress = False
        
    def check_for_updates(self):
        """Check for updates and return if an update is available"""
        try:
            app_logger.info(f"Checking for updates. Current version: {self.current_version}")
            
            # Get latest release info
            response = requests.get(UPDATE_URL, timeout=10)
            if response.status_code != 200:
                app_logger.error(f"Failed to get latest release info: {response.status_code}")
                return False, None, None
                
            release_info = response.json()
            latest_version = release_info["tag_name"].lstrip("v")
            
            # Skip version check if latest_version is not a valid semantic version
            if latest_version == "main":
                app_logger.info("Latest version is 'main', skipping version check")
                return False, None, None
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.current_version):
                app_logger.info(f"Update available: {latest_version}")
                
                # Find the zip asset
                zip_asset = None
                for asset in release_info.get("assets", []):
                    if asset["name"].endswith(".zip"):
                        zip_asset = asset
                        break
                
                if not zip_asset:
                    app_logger.error("No ZIP asset found in release")
                    return False, None, None
                
                return True, zip_asset["browser_download_url"], latest_version
            else:
                app_logger.info("No updates available")
                return False, None, None
                
        except Exception as e:
            app_logger.error(f"Error checking for updates: {e}")
            return False, None, None
    
    def start_update_process(self):
        """Start the update process in a background thread"""
        if self.update_in_progress:
            return
            
        self.update_in_progress = True
        update_thread = threading.Thread(target=self._perform_update_check)
        update_thread.daemon = True
        update_thread.start()
    
    def _perform_update_check(self):
        """Background thread to check and perform updates"""
        try:
            update_available, download_url, new_version = self.check_for_updates()
            
            if not update_available:
                self.update_in_progress = False
                return
                
            # Ask user if not in silent mode
            if not self.silent:
                def ask_user():
                    return messagebox.askyesno(
                        "Update Available", 
                        f"Version {new_version} is available. Update now?\n\n"
                        "The application will restart after updating."
                    )
                
                # Schedule dialog in main thread
                if not self.root.winfo_exists():
                    return
                    
                result = self.root.tk.call('after', 'idle', self.root.register(ask_user))
                if not result:
                    self.update_in_progress = False
                    return
            
            # Download and install update
            self._download_and_install_update(download_url, new_version)
            
        except Exception as e:
            app_logger.error(f"Update process error: {e}")
            self.update_in_progress = False
    
    def _download_and_install_update(self, download_url, new_version):
        """Download and install the update"""
        try:
            # Create temp directory
            temp_dir = Path(tempfile.mkdtemp())
            zip_path = temp_dir / "update.zip"
            extract_dir = temp_dir / "extract"
            extract_dir.mkdir(exist_ok=True)
            
            app_logger.info(f"Downloading update from {download_url}")
            
            # Show progress if not silent
            if not self.silent:
                def show_progress():
                    return messagebox.showinfo(
                        "Downloading Update", 
                        "Downloading update. The application will restart when complete."
                    )
                self.root.tk.call('after', 'idle', self.root.register(show_progress))
            
            # Download the update
            response = requests.get(download_url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            app_logger.info("Download complete. Extracting update...")
            
            # Extract the zip
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            
            app_logger.info("Update extracted. Installing...")
            
            # Determine app directory
            if getattr(sys, 'frozen', False):
                # Running from executable
                app_dir = Path(sys.executable).parent
            else:
                # Running from source
                app_dir = Path(__file__).parent.parent.parent
            
            # Create update script
            self._create_update_script(extract_dir, app_dir, new_version)
            
            # Execute the update script as a separate process
            self._execute_update_script()
            
        except Exception as e:
            app_logger.error(f"Update installation error: {e}")
            self.update_in_progress = False
            
            if not self.silent:
                self.root.tk.call('after', 'idle', self.root.register(lambda: 
                    messagebox.showerror("Update Failed", f"Update failed: {e}")
                ))
    
    def _create_update_script(self, source_dir, target_dir, new_version):
        """Create a script to perform the actual update after app closes"""
        update_script = tempfile.NamedTemporaryFile(
            prefix="update_", suffix=".bat", delete=False)
        
        script_path = update_script.name
        app_logger.info(f"Creating update script at {script_path}")
        
        # Handle the path to executable if frozen
        app_exe = sys.executable if getattr(sys, 'frozen', False) else "python run.py"
        
        with open(script_path, 'w') as f:
            f.write(f"""@echo off
echo Waiting for application to close...
timeout /t 2 /nobreak > nul

echo Updating files...
xcopy /E /I /Y "{source_dir}" "{target_dir}"

echo Updating version information...
echo {{\"version\": \"{new_version}\", \"update_date\": \"{time.strftime('%Y-%m-%d %H:%M:%S')}\"}} > "{target_dir}\\version_data.json"

echo Restarting application...
cd "{target_dir}"
start "" "{app_exe}"

echo Cleanup...
rmdir /S /Q "{source_dir.parent}"

echo Update completed!
del "%~f0"
""")
        
        self.update_script_path = script_path
    
    def _execute_update_script(self):
        """Execute the update script and exit the application"""
        try:
            # Run the update script
            subprocess.Popen([self.update_script_path], shell=True)
            
            # Exit the application after a short delay
            def exit_app():
                app_logger.info("Exiting for update to complete...")
                self.root.quit()
                sys.exit(0)
            
            self.root.after(1000, exit_app)
            
        except Exception as e:
            app_logger.error(f"Error executing update script: {e}")
            self.update_in_progress = False