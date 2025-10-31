"""
User Action Logger for DataChaEnhanced
======================================
Records all user interactions for reproducibility and fast-tracking.

This module logs:
- Tab switches
- Button clicks
- Filter settings
- Plot interactions (point selections, range selections)
- Fitting operations
- Export operations
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UserActionLogger:
    """Comprehensive logger for all user interactions."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the action logger."""
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "user_action_logs")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_session = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = None
        
        logger.info(f"UserActionLogger initialized with log directory: {self.log_dir}")
    
    def log_action(self, action_type: str, **kwargs):
        """Log a user action with metadata."""
        action = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            **kwargs
        }
        
        self.current_session.append(action)
        logger.debug(f"Logged action: {action_type} - {kwargs.get('description', '')}")
    
    def log_tab_switch(self, tab_name: str):
        """Log tab switch."""
        self.log_action(
            'tab_switch',
            tab_name=tab_name,
            description=f"Switched to {tab_name} tab"
        )
    
    def log_button_click(self, button_name: str, tab: Optional[str] = None, **kwargs):
        """Log button click."""
        self.log_action(
            'button_click',
            button_name=button_name,
            tab=tab,
            description=f"Clicked {button_name} button",
            **kwargs
        )
    
    def log_filter_change(self, filter_name: str, value: Any, tab: str = "Filters"):
        """Log filter setting change."""
        self.log_action(
            'filter_change',
            filter_name=filter_name,
            value=value,
            tab=tab,
            description=f"Changed {filter_name} filter to {value}"
        )
    
    def log_file_load(self, filepath: str, filename: str):
        """Log file loading."""
        self.current_file = filename
        self.log_action(
            'file_load',
            filepath=filepath,
            filename=filename,
            description=f"Loaded file: {filename}"
        )
    
    def log_plot_point_selection(self, 
                                 selection_type: str,  # 'linear', 'exp', 'integration'
                                 curve_type: str,      # 'hyperpol', 'depol'
                                 point_index: int,
                                 time_ms: float,
                                 value: float,
                                 **kwargs):
        """Log point selection on plot for fitting or integration."""
        self.log_action(
            'plot_point_selection',
            selection_type=selection_type,
            curve_type=curve_type,
            point_index=point_index,
            point_number=kwargs.get('point_number', point_index + 1),
            time_ms=time_ms,
            time_seconds=time_ms / 1000.0,
            value=value,
            description=f"Selected {selection_type} point {kwargs.get('point_number', point_index + 1)} for {curve_type} at {time_ms:.2f}ms, {value:.2f}pA",
            **{k: v for k, v in kwargs.items() if k != 'point_number'}
        )
    
    def log_fitting_complete(self,
                            fit_type: str,      # 'linear', 'exp'
                            curve_type: str,    # 'hyperpol', 'depol'
                            params: Dict,
                            **kwargs):
        """Log completion of a fitting operation."""
        self.log_action(
            'fitting_complete',
            fit_type=fit_type,
            curve_type=curve_type,
            parameters=params,
            description=f"Completed {fit_type} fitting for {curve_type}",
            **kwargs
        )
    
    def log_integration_range(self,
                             curve_type: str,   # 'hyperpol', 'depol'
                             start_idx: int,
                             end_idx: int,
                             start_time_ms: float,
                             end_time_ms: float,
                             integral_value: Optional[float] = None,
                             **kwargs):
        """Log integration range selection."""
        self.log_action(
            'integration_range',
            curve_type=curve_type,
            start_index=start_idx,
            end_index=end_idx,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            start_time_seconds=start_time_ms / 1000.0,
            end_time_seconds=end_time_ms / 1000.0,
            integral_value=integral_value,
            description=f"Set integration range for {curve_type}: {start_time_ms:.2f}ms - {end_time_ms:.2f}ms",
            **kwargs
        )
    
    def log_analysis_start(self, params: Dict):
        """Log start of signal analysis."""
        self.log_action(
            'analysis_start',
            parameters=params,
            description="Started signal analysis"
        )
    
    def log_analysis_complete(self, results: Optional[Dict] = None):
        """Log completion of signal analysis."""
        self.log_action(
            'analysis_complete',
            results=results,
            description="Completed signal analysis"
        )
    
    def log_export(self, export_type: str, filepath: str, **kwargs):
        """Log export operation."""
        self.log_action(
            'export',
            export_type=export_type,  # 'excel_single', 'excel_sets', 'csv', etc.
            filepath=filepath,
            description=f"Exported to {export_type}: {filepath}",
            **kwargs
        )
    
    def save_session(self, filename: Optional[str] = None) -> str:
        """Save current session to JSON file."""
        if filename is None:
            filename = f"session_{self.session_id}.json"
        
        filepath = self.log_dir / filename
        
        session_data = {
            'session_id': self.session_id,
            'current_file': self.current_file,
            'created_at': datetime.now().isoformat(),
            'actions': self.current_session,
            'action_count': len(self.current_session)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved session log to {filepath} ({len(self.current_session)} actions)")
        return str(filepath)
    
    def load_session(self, filepath: str) -> Dict:
        """Load a session from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        logger.info(f"Loaded session log from {filepath} ({session_data.get('action_count', 0)} actions)")
        return session_data
    
    def get_fitting_points(self, fit_type: str, curve_type: str) -> List[Dict]:
        """Extract fitting points from logged actions."""
        points = []
        for action in self.current_session:
            if (action.get('action_type') == 'plot_point_selection' and
                action.get('selection_type') == fit_type and
                action.get('curve_type') == curve_type):
                points.append({
                    'index': action.get('point_index'),
                    'time_ms': action.get('time_ms'),
                    'time_seconds': action.get('time_seconds'),
                    'value': action.get('value')
                })
        return sorted(points, key=lambda x: x.get('time_ms', 0))
    
    def get_integration_ranges(self) -> Dict[str, Dict]:
        """Extract integration ranges from logged actions."""
        ranges = {}
        for action in self.current_session:
            if action.get('action_type') == 'integration_range':
                curve_type = action.get('curve_type')
                ranges[curve_type] = {
                    'start_index': action.get('start_index'),
                    'end_index': action.get('end_index'),
                    'start_time_ms': action.get('start_time_ms'),
                    'end_time_ms': action.get('end_time_ms'),
                    'integral_value': action.get('integral_value')
                }
        return ranges
    
    def get_workflow_summary(self) -> Dict:
        """Get summary of workflow for fast-tracking."""
        summary = {
            'file_loaded': self.current_file,
            'filters': {},
            'analysis_params': None,
            'fitting_points': {
                'hyperpol': {'linear': [], 'exp': []},
                'depol': {'linear': [], 'exp': []}
            },
            'integration_ranges': {},
            'exports': []
        }
        
        for action in self.current_session:
            if action.get('action_type') == 'filter_change':
                summary['filters'][action.get('filter_name')] = action.get('value')
            elif action.get('action_type') == 'analysis_start':
                summary['analysis_params'] = action.get('parameters', {})
            elif action.get('action_type') == 'plot_point_selection':
                fit_type = action.get('selection_type')
                curve_type = action.get('curve_type')
                if fit_type in ['linear', 'exp']:
                    summary['fitting_points'][curve_type][fit_type].append({
                        'time_ms': action.get('time_ms'),
                        'value': action.get('value')
                    })
            elif action.get('action_type') == 'integration_range':
                curve_type = action.get('curve_type')
                summary['integration_ranges'][curve_type] = {
                    'start_time_ms': action.get('start_time_ms'),
                    'end_time_ms': action.get('end_time_ms')
                }
            elif action.get('action_type') == 'export':
                summary['exports'].append({
                    'type': action.get('export_type'),
                    'filepath': action.get('filepath')
                })
        
        return summary
    
    def clear_session(self):
        """Clear current session."""
        self.current_session = []
        self.current_file = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Cleared current session")

