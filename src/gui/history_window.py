import tkinter as tk
from tkinter import ttk
from src.utils.logger import app_logger

class HistoryWindow(tk.Toplevel):
    def __init__(self, parent, history_manager):
        """Initialize the history window."""
        super().__init__(parent)
        self.title("Analysis History")
        self.history_manager = history_manager
        
        # Set window size and position
        window_width = 800
        window_height = 600
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        self.setup_ui()
        self.load_history()

    def setup_ui(self):
        """Setup the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(fill='both', expand=True)

        # Create treeview
        columns = (
            'timestamp', 'filename', 'integral_value', 
            'hyperpol_area', 'depol_area', 'linear_capacitance',
            'v2_voltage'
        )
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        
        # Configure scrollbars
        y_scroll = ttk.Scrollbar(main_frame, orient='vertical', command=self.tree.yview)
        x_scroll = ttk.Scrollbar(main_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Setup column headings
        column_names = {
            'timestamp': 'Timestamp',
            'filename': 'Filename',
            'integral_value': 'Integral Value',
            'hyperpol_area': 'Hyperpol Area',
            'depol_area': 'Depol Area',
            'linear_capacitance': 'Linear Capacitance',
            'v2_voltage': 'V2 Voltage'
        }
        
        for col in columns:
            self.tree.heading(col, text=column_names[col],
                            command=lambda c=col: self.sort_column(c))
            self.tree.column(col, width=100, minwidth=50)

        # Pack components
        self.tree.pack(side='left', fill='both', expand=True)
        y_scroll.pack(side='right', fill='y')
        x_scroll.pack(side='bottom', fill='x')
        
        # Add control buttons
        button_frame = ttk.Frame(self, padding="5")
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Refresh",
                  command=self.load_history).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear History",
                  command=self.clear_history).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close",
                  command=self.destroy).pack(side='right', padx=5)

    def load_history(self):
        """Load and display history entries."""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Load history entries
            history = self.history_manager.get_history()
            
            # Insert entries into treeview
            for entry in history:
                values = (
                    entry['timestamp'],
                    entry['filename'],
                    entry['integral_value'],
                    entry['hyperpol_area'],
                    entry['depol_area'],
                    entry['linear_capacitance'],
                    entry['v2_voltage']
                )
                self.tree.insert('', 'end', values=values)
                
            app_logger.info("History loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Error loading history: {str(e)}")

    def clear_history(self):
        """Clear all history entries."""
        try:
            self.history_manager.clear_history()
            self.load_history()
            app_logger.info("History cleared")
        except Exception as e:
            app_logger.error(f"Error clearing history: {str(e)}")

    def sort_column(self, col):
        """Sort treeview by selected column."""
        try:
            # Get all items
            items = [(self.tree.set(item, col), item) for item in self.tree.get_children('')]
            
            # Sort items
            items.sort(reverse=getattr(self, "sort_reverse", False))
            
            # Rearrange items in sorted positions
            for index, (_, item) in enumerate(items):
                self.tree.move(item, '', index)
            
            # Switch sort order for next click
            self.sort_reverse = not getattr(self, "sort_reverse", False)
            
        except Exception as e:
            app_logger.error(f"Error sorting column: {str(e)}")