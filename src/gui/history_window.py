"""
History window module for displaying analysis history.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger

class HistoryWindow:
    """
    Window for displaying the analysis history.
    """
    def __init__(self, parent, history_manager):
        """
        Initialize the history window.
        
        Args:
            parent: Parent widget
            history_manager: The AnalysisHistoryManager instance
        """
        self.parent = parent
        self.history_manager = history_manager
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Analysis History")
        self.window.geometry("800x400")
        self.window.transient(parent)
        self.window.grab_set()
        
        # Set icon if available
        if hasattr(parent, 'iconbitmap'):
            icon_path = os.path.join(os.path.dirname(__file__), "../../assets/icon.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        
        # Create treeview for history entries
        self.columns = (
            "timestamp", "filename", "integral_value", 
            "hyperpol_area", "depol_area", "capacitance_nF", "v2_voltage"
        )
        self.column_headings = {
            "timestamp": "Timestamp",
            "filename": "Filename",
            "integral_value": "Integral Value",
            "hyperpol_area": "Hyperpol Area",
            "depol_area": "Depol Area",
            "capacitance_nF": "Linear Capacitance",
            "v2_voltage": "V2 Voltage"
        }
        
        self.create_widgets()
        self.populate_tree()
    
    def create_widgets(self):
        """Create and place widgets in the window."""
        # Create treeview
        self.tree = ttk.Treeview(
            self.window, 
            columns=self.columns,
            show='headings',
            selectmode='browse'
        )
        
        # Set column headings and widths
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.column("timestamp", width=150, anchor="w")
        
        self.tree.heading("filename", text="Filename")
        self.tree.column("filename", width=200, anchor="w")
        
        self.tree.heading("integral_value", text="Integral Value")
        self.tree.column("integral_value", width=100, anchor="center")
        
        self.tree.heading("hyperpol_area", text="Hyperpol Area")
        self.tree.column("hyperpol_area", width=100, anchor="center")
        
        self.tree.heading("depol_area", text="Depol Area")
        self.tree.column("depol_area", width=100, anchor="center")
        
        self.tree.heading("capacitance_nF", text="Linear Capacitance")
        self.tree.column("capacitance_nF", width=120, anchor="center")
        
        self.tree.heading("v2_voltage", text="V2 Voltage")
        self.tree.column("v2_voltage", width=80, anchor="center")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.window, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.tree.pack(side='top', fill='both', expand=True)
        
        # Add buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
        
        refresh_btn = ttk.Button(
            button_frame, 
            text="Refresh", 
            command=self.refresh
        )
        refresh_btn.pack(side='left', padx=5)
        
        clear_btn = ttk.Button(
            button_frame, 
            text="Clear History", 
            command=self.clear_history
        )
        clear_btn.pack(side='left', padx=5)
        
        close_btn = ttk.Button(
            button_frame, 
            text="Close", 
            command=self.window.destroy
        )
        close_btn.pack(side='right', padx=5)
    
    def populate_tree(self):
        """Populate the treeview with history entries."""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # Filter to show only manual analyses or the most recent analysis for each file
            history_entries = self.history_manager.history_entries
            
            # Get unique filenames
            filenames = set(entry['filename'] for entry in history_entries)
            
            # For each unique filename, get the most recent entry
            filtered_entries = []
            for filename in filenames:
                file_entries = [entry for entry in history_entries if entry['filename'] == filename]
                # Sort by timestamp (newest first)
                file_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                # Add the most recent entry
                filtered_entries.append(file_entries[0])
            
            # Sort by timestamp (newest first)
            filtered_entries.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Add entries to tree
            for entry in filtered_entries:
                self.tree.insert(
                    '', 'end', 
                    values=(
                        entry.get('timestamp', ''),
                        entry.get('filename', ''),
                        entry.get('integral_value', ''),
                        entry.get('hyperpol_area', ''),
                        entry.get('depol_area', ''),
                        entry.get('capacitance_nF', ''),
                        entry.get('v2_voltage', '')
                    )
                )
            
            app_logger.debug(f"Populated history tree with {len(filtered_entries)} entries")
                
        except Exception as e:
            app_logger.error(f"Error populating history tree: {str(e)}")
            messagebox.showerror("Error", f"Failed to display history: {str(e)}")
    
    def refresh(self):
        """Refresh the tree view with current history data."""
        self.populate_tree()
        app_logger.debug("History display refreshed")
    
    def clear_history(self):
        """Clear all history entries."""
        confirm = messagebox.askyesno(
            "Confirm Clear", 
            "Are you sure you want to clear all history entries?"
        )
        
        if confirm:
            self.history_manager.history_entries = []
            self.populate_tree()
            app_logger.info("Analysis history cleared")