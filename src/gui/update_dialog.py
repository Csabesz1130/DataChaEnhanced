"""
Update dialog for Signal Analyzer
Shows when a new version is available
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UpdateDialog:
    """
    Dialog that prompts user about available updates
    
    Usage:
        update_info = updater.check_for_updates()
        if update_info['available']:
            dialog = UpdateDialog(root, update_info)
    """
    
    def __init__(self, parent: tk.Tk, update_info: Dict):
        """
        Initialize update dialog
        
        Args:
            parent: Parent Tkinter window
            update_info: Dictionary with update information from AutoUpdater
        """
        self.parent = parent
        self.update_info = update_info
        self.updater = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Update Available")
        self.dialog.geometry("600x450")
        self.dialog.resizable(False, False)
        
        # Center on parent
        self._center_on_parent()
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create UI
        self._create_widgets()
        
        logger.info(f"Update dialog shown for version {update_info['version']}")
    
    def _center_on_parent(self):
        """Center dialog on parent window"""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _create_widgets(self):
        """Create all UI widgets"""
        
        # Header
        header_frame = tk.Frame(self.dialog, bg='#4CAF50', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🎉 Update Available!",
            font=('Arial', 16, 'bold'),
            bg='#4CAF50',
            fg='white'
        ).pack(pady=10)
        
        version_text = f"Version {self.update_info['version']} is ready to install"
        tk.Label(
            header_frame,
            text=version_text,
            font=('Arial', 10),
            bg='#4CAF50',
            fg='white'
        ).pack()
        
        # Main content
        content_frame = tk.Frame(self.dialog, padx=20, pady=20)
        content_frame.pack(fill='both', expand=True)
        
        # Size information
        size_mb = self.update_info.get('size', 0) / (1024 * 1024)
        info_text = f"Download size: {size_mb:.1f} MB"
        tk.Label(
            content_frame,
            text=info_text,
            font=('Arial', 9),
            fg='gray'
        ).pack(anchor='w', pady=(0, 10))
        
        # Changelog section
        tk.Label(
            content_frame,
            text="What's New:",
            font=('Arial', 11, 'bold')
        ).pack(anchor='w', pady=(0, 5))
        
        # Changelog text with scrollbar
        changelog_frame = tk.Frame(content_frame)
        changelog_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(changelog_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.changelog_text = tk.Text(
            changelog_frame,
            height=10,
            wrap='word',
            font=('Arial', 9),
            yscrollcommand=scrollbar.set,
            relief='solid',
            borderwidth=1
        )
        self.changelog_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.changelog_text.yview)
        
        # Insert changelog
        changelog = self.update_info.get('changelog', 'No changelog available')
        self.changelog_text.insert('1.0', changelog)
        self.changelog_text.config(state='disabled')
        
        # Progress section (initially hidden)
        self.progress_frame = tk.Frame(content_frame)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Preparing download...",
            font=('Arial', 9)
        )
        self.progress_label.pack(anchor='w', pady=(5, 5))
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill='x')
        
        self.progress_detail = tk.Label(
            self.progress_frame,
            text="",
            font=('Arial', 8),
            fg='gray'
        )
        self.progress_detail.pack(anchor='w', pady=(2, 0))
        
        # Buttons
        button_frame = tk.Frame(self.dialog, padx=20, pady=20)
        button_frame.pack(fill='x')
        
        # Update button
        self.update_btn = tk.Button(
            button_frame,
            text="📥 Update Now",
            command=self._start_update,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=8,
            relief='flat',
            cursor='hand2'
        )
        self.update_btn.pack(side='left', padx=5)
        
        # View on GitHub button
        self.github_btn = tk.Button(
            button_frame,
            text="🔗 View on GitHub",
            command=self._open_github,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10),
            padx=20,
            pady=8,
            relief='flat',
            cursor='hand2'
        )
        self.github_btn.pack(side='left', padx=5)
        
        # Later button
        self.later_btn = tk.Button(
            button_frame,
            text="Remind Me Later",
            command=self._remind_later,
            font=('Arial', 10),
            padx=20,
            pady=8,
            relief='flat',
            cursor='hand2'
        )
        self.later_btn.pack(side='right', padx=5)
    
    def _start_update(self):
        """Start the update download and installation process"""
        logger.info("User initiated update download")
        
        # Disable buttons
        self.update_btn.config(state='disabled')
        self.github_btn.config(state='disabled')
        self.later_btn.config(state='disabled')
        
        # Show progress
        self.progress_frame.pack(fill='x', pady=(10, 0))
        
        # Start download in background thread
        thread = threading.Thread(target=self._download_and_apply, daemon=True)
        thread.start()
    
    def _download_and_apply(self):
        """Download and apply update (runs in background thread)"""
        try:
            # Import here to avoid circular imports
            from src.utils.updater import AutoUpdater
            from src.utils.logger import app_logger
            
            # Get current version
            try:
                with open('version_info.json', 'r') as f:
                    import json
                    version_data = json.load(f)
                    current_version = version_data.get('version', '1.0.0')
            except:
                current_version = '1.0.0'
            
            # Create updater
            self.updater = AutoUpdater(current_version, "yourusername/DataChaEnhanced")
            
            # Download with progress
            def progress_callback(downloaded, total):
                percent = (downloaded / total * 100) if total > 0 else 0
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                
                # Update UI (must use after() to update from thread)
                self.dialog.after(0, self._update_progress, percent, downloaded_mb, total_mb)
            
            self.dialog.after(0, self._update_status, "Downloading update...")
            
            update_file = self.updater.download_update(
                self.update_info['download_url'],
                progress_callback
            )
            
            if not update_file:
                self.dialog.after(0, self._download_failed, "Download failed")
                return
            
            # Verify download
            self.dialog.after(0, self._update_status, "Verifying download...")
            if not self.updater.verify_download(update_file):
                self.dialog.after(0, self._download_failed, "Download verification failed")
                return
            
            # Apply update
            self.dialog.after(0, self._update_status, "Installing update...")
            if self.updater.apply_update(update_file, backup=True):
                self.dialog.after(0, self._update_success)
            else:
                self.dialog.after(0, self._download_failed, "Failed to apply update")
        
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=True)
            self.dialog.after(0, self._download_failed, str(e))
    
    def _update_progress(self, percent: float, downloaded_mb: float, total_mb: float):
        """Update progress bar (called from main thread)"""
        self.progress_bar['value'] = percent
        self.progress_detail.config(
            text=f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB ({percent:.0f}%)"
        )
    
    def _update_status(self, status: str):
        """Update status label (called from main thread)"""
        self.progress_label.config(text=status)
    
    def _update_success(self):
        """Handle successful update"""
        logger.info("Update downloaded and ready to install")
        
        messagebox.showinfo(
            "Update Ready",
            "Update has been downloaded successfully!\n\n"
            "The application will now restart to complete the update.",
            parent=self.dialog
        )
        
        # Close application to allow updater to run
        self.dialog.destroy()
        self.parent.quit()
    
    def _download_failed(self, error: str):
        """Handle download failure"""
        logger.error(f"Update download failed: {error}")
        
        # Re-enable buttons
        self.update_btn.config(state='normal')
        self.github_btn.config(state='normal')
        self.later_btn.config(state='normal')
        
        # Hide progress
        self.progress_frame.pack_forget()
        
        # Show error
        messagebox.showerror(
            "Update Failed",
            f"Failed to download or install update:\n{error}\n\n"
            "You can try again later or download manually from GitHub.",
            parent=self.dialog
        )
    
    def _open_github(self):
        """Open GitHub releases page in browser"""
        url = self.update_info.get('html_url', 
                                   'https://github.com/yourusername/DataChaEnhanced/releases')
        webbrowser.open(url)
        logger.info(f"Opened GitHub releases: {url}")
    
    def _remind_later(self):
        """Close dialog and remind later"""
        logger.info("User chose to be reminded later")
        self.dialog.destroy()


def show_update_dialog(parent: tk.Tk, update_info: Dict):
    """
    Convenience function to show update dialog
    
    Args:
        parent: Parent Tkinter window
        update_info: Update information from AutoUpdater
    """
    if update_info.get('available'):
        UpdateDialog(parent, update_info)
        return True
    return False


if __name__ == "__main__":
    # Test the update dialog
    root = tk.Tk()
    root.title("Signal Analyzer")
    root.geometry("800x600")
    
    # Fake update info for testing
    test_update_info = {
        'available': True,
        'version': '1.0.3',
        'download_url': 'https://github.com/test/test/releases/download/v1.0.3/SignalAnalyzer.zip',
        'changelog': """### New Features
- Added automatic update functionality
- Improved startup performance
- Enhanced error handling

### Bug Fixes
- Fixed crash when loading large files
- Fixed memory leak in plotting
- Fixed incorrect calculations in some edge cases

### Improvements
- Faster file loading (up to 2x speedup)
- Better UI responsiveness
- Reduced memory usage""",
        'size': 50 * 1024 * 1024,  # 50 MB
        'html_url': 'https://github.com/test/test/releases/tag/v1.0.3'
    }
    
    # Show dialog after 1 second
    root.after(1000, lambda: show_update_dialog(root, test_update_info))
    
    root.mainloop()

