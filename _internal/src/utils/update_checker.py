# src/utils/update_checker.py
import webbrowser
import tkinter as tk
from tkinter import messagebox
import threading
import requests
import json
from packaging import version
from src.version import APP_VERSION, GITHUB_REPO
from src.utils.logger import app_logger

def check_for_updates(show_current_dialog=False):
    """
    Check GitHub for newer releases
    
    Args:
        show_current_dialog (bool): Whether to show "up to date" dialog
        
    Returns:
        tuple: (is_update_available, download_url, latest_version)
    """
    try:
        # Get latest release info
        response = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            timeout=5
        )
        release_info = json.loads(response.text)
        latest_version = release_info["tag_name"].lstrip("v")
        
        app_logger.info(f"Current version: {APP_VERSION}, Latest: {latest_version}")
        
        if version.parse(latest_version) > version.parse(APP_VERSION):
            download_url = release_info["html_url"]
            return True, download_url, latest_version
        elif show_current_dialog:
            messagebox.showinfo(
                "No Updates Available", 
                f"You are running the latest version ({APP_VERSION})."
            )
        return False, None, None
    except Exception as e:
        app_logger.error(f"Update check failed: {str(e)}")
        if show_current_dialog:
            messagebox.showwarning(
                "Update Check Failed", 
                f"Could not check for updates: {str(e)}"
            )
        return False, None, None

def check_updates_background(root):
    """Run update check in background thread to avoid freezing UI"""
    def _bg_check():
        update_available, download_url, new_version = check_for_updates()
        if update_available:
            # Schedule dialog in main thread
            root.after(0, lambda: _show_update_dialog(download_url, new_version))
    
    # Start background thread
    thread = threading.Thread(target=_bg_check)
    thread.daemon = True
    thread.start()

def _show_update_dialog(download_url, new_version):
    """Show update notification dialog"""
    answer = messagebox.askyesno(
        "Update Available", 
        f"Version {new_version} is available (you have {APP_VERSION}).\n\n"
        f"Would you like to download the new version now?"
    )
    if answer:
        webbrowser.open(download_url)