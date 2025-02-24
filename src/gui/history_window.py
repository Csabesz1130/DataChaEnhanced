import os
import datetime
import tkinter as tk
from tkinter import ttk
from src.utils.logger import app_logger

class AnalysisHistoryManager:
    """
    Manages the history of analyses performed in the application.
    Stores and displays analysis results with timestamps.
    """
    def __init__(self, parent):
        self.parent = parent
        self.history_entries = []
        self.history_window = None
        
    def add_entry(self, filename, results):
        """
        Add a new entry to the analysis history.
        
        Args:
            filename (str): The name of the analyzed file
            results (dict): The analysis results including integral values
        """
        try:
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract relevant data from results
            integral_value = results.get('integral_value', 'N/A')
            if isinstance(integral_value, str) and ',' in integral_value:
                integral_value = integral_value.split(',')[0].strip()
                
            hyperpol_area = results.get('hyperpol_area', 'N/A')
            depol_area = results.get('depol_area', 'N/A')
            capacitance = results.get('capacitance_nF', 'N/A')
            
            # Get V2 value
            v2_voltage = "N/A"
            if hasattr(self.parent, 'action_potential_processor'):
                processor = self.parent.action_potential_processor
                if processor and hasattr(processor, 'params'):
                    v2_voltage = f"{processor.params.get('V2', 'N/A')} mV"
            
            # Create the entry
            entry = {
                'timestamp': timestamp,
                'filename': os.path.basename(filename) if filename else "Unknown",
                'integral_value': integral_value,
                'hyperpol_area': hyperpol_area,
                'depol_area': depol_area,
                'capacitance_nF': capacitance,
                'v2_voltage': v2_voltage
            }
            
            # Add to history
            self.history_entries.append(entry)
            app_logger.info(f"Added history entry: {entry['filename']}, {entry['integral_value']}")
            
        except Exception as e:
            app_logger.error(f"Error adding history entry: {str(e)}")
    
    def show_history_dialog(self):
        """Show dialog with analysis history."""
        if not self.history_entries:
            tk.messagebox.showinfo("History", "No analysis history available.")
            return
            
        if self.history_window and self.history_window.winfo_exists():
            self.history_window.focus_force()
            self.refresh_history_display()
            return
            
        try:
            # Create new window
            self.history_window = tk.Toplevel(self.parent.master)
            self.history_window.title("Analysis History")
            self.history_window.geometry("800x400")
            self.history_window.transient(self.parent.master)
            self.history_window.grab_set()
            
            # Set icon
            if hasattr(self.parent.master, 'iconbitmap'):
                icon_path = os.path.join(os.path.dirname(__file__), "../../assets/icon.ico")
                if os.path.exists(icon_path):
                    self.history_window.iconbitmap(icon_path)
            
            # Create treeview for history entries
            columns = (
                "timestamp", "filename", "integral_value", 
                "hyperpol_area", "depol_area", "capacitance_nF", "v2_voltage"
            )
            column_headings = {
                "timestamp": "Timestamp",
                "filename": "Filename",
                "integral_value": "Integral Value",
                "hyperpol_area": "Hyperpol Area",
                "depol_area": "Depol Area",
                "capacitance_nF": "Linear Capacitance",
                "v2_voltage": "V2 Voltage"
            }
            
            tree = ttk.Treeview(
                self.history_window, 
                columns=columns,
                show='headings',
                selectmode='browse'
            )
            
            # Set column headings and widths
            tree.heading("timestamp", text="Timestamp")
            tree.column("timestamp", width=150, anchor="w")
            
            tree.heading("filename", text="Filename")
            tree.column("filename", width=200, anchor="w")
            
            tree.heading("integral_value", text="Integral Value")
            tree.column("integral_value", width=100, anchor="center")
            
            tree.heading("hyperpol_area", text="Hyperpol Area")
            tree.column("hyperpol_area", width=100, anchor="center")
            
            tree.heading("depol_area", text="Depol Area")
            tree.column("depol_area", width=100, anchor="center")
            
            tree.heading("capacitance_nF", text="Linear Capacitance")
            tree.column("capacitance_nF", width=120, anchor="center")
            
            tree.heading("v2_voltage", text="V2 Voltage")
            tree.column("v2_voltage", width=80, anchor="center")
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(self.history_window, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y')
            tree.pack(side='top', fill='both', expand=True)
            
            # Add buttons
            button_frame = ttk.Frame(self.history_window)
            button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
            
            refresh_btn = ttk.Button(
                button_frame, 
                text="Refresh", 
                command=self.refresh_history_display
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
                command=self.history_window.destroy
            )
            close_btn.pack(side='right', padx=5)
            
            # Store reference to tree
            self.history_tree = tree
            
            # Fill tree with data
            self.refresh_history_display()
            
        except Exception as e:
            app_logger.error(f"Error showing history dialog: {str(e)}")
    
    def refresh_history_display(self):
        """Refresh the history display with current entries."""
        if not hasattr(self, 'history_tree') or not self.history_tree.winfo_exists():
            return
            
        try:
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
                
            # Add history entries
            for entry in reversed(self.history_entries):  # Newest first
                self.history_tree.insert(
                    '', 'end', 
                    values=(
                        entry['timestamp'],
                        entry['filename'],
                        entry['integral_value'],
                        entry['hyperpol_area'],
                        entry['depol_area'],
                        entry['capacitance_nF'],
                        entry['v2_voltage']
                    )
                )
                
        except Exception as e:
            app_logger.error(f"Error refreshing history display: {str(e)}")
    
    def clear_history(self):
        """Clear the analysis history."""
        self.history_entries = []
        self.refresh_history_display()
        app_logger.info("Analysis history cleared")