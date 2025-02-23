import os
import json
from datetime import datetime
from src.utils.logger import app_logger

class HistoryManager:
    def __init__(self):
        """Initialize the history manager."""
        self.history_file = "analysis_history.json"
        self.history = self.load_history()

    def load_history(self):
        """Load history from JSON file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            app_logger.error(f"Error loading history: {str(e)}")
            return []

    def save_history(self):
        """Save history to JSON file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            app_logger.info("History saved successfully")
        except Exception as e:
            app_logger.error(f"Error saving history: {str(e)}")

    def add_entry(self, filename, results):
        """
        Add a new entry to the history.
        
        Args:
            filename (str): Name of the analyzed file
            results (dict): Analysis results including integral values
        """
        try:
            # Extract relevant values
            entry = {
                'filename': os.path.basename(filename),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'integral_value': results.get('integral_value', 'N/A'),
                'hyperpol_area': results.get('hyperpol_area', 'N/A'),
                'depol_area': results.get('depol_area', 'N/A'),
                'linear_capacitance': results.get('capacitance_nF', 'N/A'),
                'v2_voltage': results.get('v2_voltage', 'N/A')
            }

            # Add to history list
            self.history.append(entry)
            
            # Keep only last 100 entries
            if len(self.history) > 100:
                self.history = self.history[-100:]
                
            # Save updated history
            self.save_history()
            app_logger.info(f"Added history entry for {filename}")
            
        except Exception as e:
            app_logger.error(f"Error adding history entry: {str(e)}")

    def get_history(self):
        """Get the complete history list."""
        return self.history

    def clear_history(self):
        """Clear all history entries."""
        try:
            self.history = []
            self.save_history()
            app_logger.info("History cleared")
        except Exception as e:
            app_logger.error(f"Error clearing history: {str(e)}")