"""
Point counter utility module for the Signal Analyzer application.
Provides functionality to track and display cursor position on multiple curves.
"""

import numpy as np
from matplotlib.text import Text
from src.utils.logger import app_logger

class CurvePointTracker:
    """
    Tracks cursor position on multiple curves and displays point information.
    Supports normalized (blue), averaged (magenta), and purple curves.
    """
    
    def __init__(self, figure, ax, status_var=None):
        """
        Initialize the curve point tracker.
        
        Args:
            figure: Matplotlib figure
            ax: Matplotlib axes
            status_var: Optional Tkinter StringVar for status display
        """
        self.fig = figure
        self.ax = ax
        self.status_var = status_var  # New parameter for status bar
        self.annotations = {}
        self.curve_data = {
            'orange': {'data': None, 'times': None, 'visible': False},
            'blue': {'data': None, 'times': None, 'visible': False},
            'magenta': {'data': None, 'times': None, 'visible': False},
            'purple_hyperpol': {'data': None, 'times': None, 'visible': False},
            'purple_depol': {'data': None, 'times': None, 'visible': False}
        }
        self.show_points = False
        self.last_cursor_pos = None
        self.current_time = 0
        self.current_value = 0
        
        # Text position offsets for each curve to prevent overlap
        self.offsets = {
            'orange': (10, 10),
            'blue': (10, 30),
            'magenta': (10, 50),
            'purple_hyperpol': (10, 70),
            'purple_depol': (10, 90)
        }
        
        # Color mapping for annotation text
        self.colors = {
            'orange': 'orange',
            'blue': 'blue',
            'magenta': 'magenta',
            'purple_hyperpol': 'purple',
            'purple_depol': 'darkviolet'
        }
        
        # Display names for the curves
        self.curve_names = {
            'orange': 'Orange',
            'blue': 'Blue',
            'magenta': 'Magenta',
            'purple_hyperpol': 'Purple Hyperpol',
            'purple_depol': 'Purple Depol'
        }
        
        # Setup event connections
        self._connect()
        
    def _connect(self):
        """Connect to matplotlib event callbacks"""
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.cid_figure = self.fig.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)
    
    def _disconnect(self):
        """Disconnect from matplotlib event callbacks"""
        self.fig.canvas.mpl_disconnect(self.cid_move)
        self.fig.canvas.mpl_disconnect(self.cid_figure)
    
    def set_curve_data(self, curve_type, data, times=None, visible=True):
        """
        Set data for a specific curve type.
        
        Args:
            curve_type: Type of curve ('orange', 'blue', 'magenta', 'purple_hyperpol', 'purple_depol')
            data: Array of y-values
            times: Array of x-values (timestamps)
            visible: Whether the curve is visible
        """
        if curve_type in self.curve_data:
            self.curve_data[curve_type]['data'] = data
            self.curve_data[curve_type]['times'] = times
            self.curve_data[curve_type]['visible'] = visible
            app_logger.debug(f"Set {curve_type} curve data with {len(data) if data is not None else 0} points")
    
    def set_show_points(self, show):
        """
        Set whether to show points.
        
        Args:
            show: Boolean flag to show or hide points
        """
        self.show_points = show
        
        # Clear annotations if points are hidden
        if not show:
            self.clear_annotations()
            if self.status_var:
                self._clear_status_display()
    
    def clear_annotations(self):
        """Remove all point annotations"""
        for ann in self.annotations.values():
            if ann is not None:
                ann.remove()
        self.annotations = {}
        self.fig.canvas.draw_idle()
    
    def _clear_status_display(self):
        """Clear the status display"""
        if self.status_var:
            self.status_var.set("")
    
    def _get_nearest_point(self, x, y, curve_type):
        """
        Find the nearest point on a curve to the cursor position.
        Enhanced version with better threshold logic.
        
        Args:
            x: Cursor x-position
            y: Cursor y-position
            curve_type: Type of curve
            
        Returns:
            Tuple of (index, distance, x_val, y_val) or None if no point found
        """
        curve_data = self.curve_data[curve_type]
        if curve_data['data'] is None or not curve_data['visible']:
            return None
            
        data = curve_data['data']
        times = curve_data['times']
        
        if times is None:
            # If no times provided, use indices
            times = np.arange(len(data))
        
        # Convert axes coordinates to data coordinates
        data_x, data_y = self.ax.transData.inverted().transform((x, y))
        
        # Find closest point by x-value first
        idx = np.abs(times - data_x).argmin()
        
        # Calculate distance to point
        x_distance = abs(times[idx] - data_x)
        y_distance = abs(data[idx] - data_y)
        
        # Get axis ranges for proper distance scaling
        x_range = np.max(times) - np.min(times)
        y_range = np.max(data) - np.min(data)
        
        # Scale distances by axis ranges for proper weighting
        # This makes the threshold more robust regardless of units
        scaled_x_distance = x_distance / (x_range * 0.05)  # 5% of x-range
        scaled_y_distance = y_distance / (y_range * 0.1)   # 10% of y-range
        
        # Use weighted distance - prioritize x-distance with 2:1 ratio
        weighted_distance = np.sqrt((2 * scaled_x_distance)**2 + scaled_y_distance**2)
        
        # Return the point if it's close enough (distance < 1.0 means within threshold)
        if weighted_distance < 1.0:
            return (idx, weighted_distance, times[idx], data[idx])
        
        return None
    
    def _get_corresponding_orange_point(self, curve_type, point_idx):
        """
        Get the corresponding orange curve point index.
        
        Args:
            curve_type: Type of curve
            point_idx: Index of the point on the curve
            
        Returns:
            Corresponding orange point index or None
        """
        # Return None if orange curve data is not available
        if self.curve_data['orange']['data'] is None:
            return None
            
        # For normalized (blue) curve, the corresponding index is direct
        if curve_type == 'blue':
            # Ensure the index is within range of orange curve
            if point_idx < len(self.curve_data['orange']['data']):
                return point_idx
        
        # For averaged (magenta) curve, calculate scaled index
        elif curve_type == 'magenta':
            # We need to map the magenta index to original data
            # This is an approximation - in real implementation, you might 
            # need a more accurate mapping based on your data structure
            orange_len = len(self.curve_data['orange']['data'])
            magenta_len = len(self.curve_data[curve_type]['data'])
            
            if magenta_len > 0 and orange_len > 0:
                # Scale the index proportionally
                scaled_idx = int(point_idx * orange_len / magenta_len)
                return min(scaled_idx, orange_len - 1)
        
        # For purple curves, use domain knowledge to map indices
        elif curve_type in ['purple_hyperpol', 'purple_depol']:
            # This would depend on your specific implementation
            # For instance, if purple curves are extracted segments,
            # you might have stored start/end indices
            processor = getattr(self.ax.figure.canvas.manager.window, 'action_potential_processor', None)
            
            if processor is not None:
                if curve_type == 'purple_hyperpol' and hasattr(processor, 'hyperpol_indices'):
                    start_idx = processor.hyperpol_indices[0]
                    return start_idx + point_idx
                
                elif curve_type == 'purple_depol' and hasattr(processor, 'depol_indices'):
                    start_idx = processor.depol_indices[0]
                    return start_idx + point_idx
        
        return None
    
    def _on_mouse_move(self, event):
        """
        Handle mouse movement events.
        Enhanced to update both annotations and status bar.
        
        Args:
            event: Matplotlib motion event
        """
        if not event.inaxes or not self.show_points or event.inaxes != self.ax:
            # Clear status bar if cursor is not over the plot
            if self.status_var:
                self._clear_status_display()
            return
            
        # Store cursor position and data values
        self.last_cursor_pos = (event.x, event.y)
        self.current_time = event.xdata
        self.current_value = event.ydata
        
        # For status bar: Find the closest point across all curves
        closest_curve = None
        closest_info = None
        min_distance = float('inf')
        
        # Define the order of priority for curves (in case of ties)
        curve_priority = ['orange', 'blue', 'magenta', 'purple_hyperpol', 'purple_depol']
        
        # First, check each curve and collect all points within threshold
        candidate_points = []
        
        for curve_type in curve_priority:
            if self.curve_data[curve_type]['visible']:
                point_info = self._get_nearest_point(event.x, event.y, curve_type)
                
                if point_info is not None:
                    idx, distance, x_val, y_val = point_info
                    candidate_points.append({
                        'curve_type': curve_type,
                        'distance': distance,
                        'index': idx,
                        'x_val': x_val,
                        'y_val': y_val
                    })
        
        # If we have candidate points, find the closest one for status bar
        if candidate_points:
            # Sort by distance
            candidate_points.sort(key=lambda p: p['distance'])
            
            # Get the closest point
            closest = candidate_points[0]
            closest_curve = closest['curve_type']
            closest_info = {
                'index': closest['index'],
                'x_val': closest['x_val'],
                'y_val': closest['y_val']
            }
        
        # Update status bar based on closest point
        if self.status_var and closest_curve is not None and closest_info is not None:
            self._update_status_for_curve(closest_curve, closest_info)
        
        # Process each curve type for annotations (original behavior)
        for curve_type in self.curve_data:
            # Skip if curve is not visible
            if not self.curve_data[curve_type]['visible']:
                if curve_type in self.annotations and self.annotations[curve_type] is not None:
                    self.annotations[curve_type].remove()
                    self.annotations[curve_type] = None
                continue
                
            # Get nearest point
            nearest = self._get_nearest_point(event.x, event.y, curve_type)
            
            if nearest is not None:
                # Get point information
                idx, dist, x_val, y_val = nearest
                
                # Get corresponding orange point
                orange_idx = self._get_corresponding_orange_point(curve_type, idx)
                
                # Create annotation text with orange point reference
                color_name = curve_type.split('_')[0].capitalize()
                if '_' in curve_type:
                    color_name += f" {curve_type.split('_')[1].capitalize()}"
                
                if orange_idx is not None:
                    text = f"{color_name} point: {idx} [Orange: {orange_idx}]"
                else:
                    text = f"{color_name} point: {idx} [No Orange ref]"
                
                # Add or update annotation
                if curve_type in self.annotations and self.annotations[curve_type] is not None:
                    self.annotations[curve_type].set_text(text)
                    self.annotations[curve_type].set_position((x_val, y_val))
                else:
                    offset_x, offset_y = self.offsets[curve_type]
                    self.annotations[curve_type] = self.ax.annotate(
                        text, xy=(x_val, y_val),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        color=self.colors[curve_type],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color=self.colors[curve_type])
                    )
            elif curve_type in self.annotations and self.annotations[curve_type] is not None:
                self.annotations[curve_type].remove()
                self.annotations[curve_type] = None
        
        # Redraw canvas
        self.fig.canvas.draw_idle()
    
    def _update_status_for_curve(self, curve_type, point_info):
        """
        Update status display for a specific curve.
        
        Args:
            curve_type: Type of curve
            point_info: Dictionary with point information
        """
        if not self.status_var:
            return
            
        try:
            # Get time and current value at cursor position
            time_str = f"Time: {self.current_time:.1f} ms"
            current_str = f"Current: {self.current_value:.1f} pA"
            
            # Get point information
            curve_name = self.curve_names[curve_type]
            point_idx = point_info['index']
            
            # Create the point display text - ensure it matches the expected format
            point_str = f"{curve_name} Point: {point_idx}"
            
            # Add orange reference for non-orange curves
            if curve_type != 'orange':
                orange_idx = self._get_corresponding_orange_point(curve_type, point_idx)
                if orange_idx is not None:
                    # Use the exact format requested: "Orange Point: X" 
                    point_str += f" (Orange Point: {orange_idx})"
                else:
                    point_str += " (No Orange ref)"
            
            # Combine all parts
            status_text = f"{time_str}, {current_str}, {point_str}"
            self.status_var.set(status_text)
            
        except Exception as e:
            app_logger.error(f"Error updating status: {str(e)}")
            # Fallback to basic status
            self._update_basic_status()
    
    def _update_basic_status(self):
        """Update status bar with just time and current value"""
        if self.status_var:
            status_text = f"Time: {self.current_time:.1f} ms, Current: {self.current_value:.1f} pA"
            self.status_var.set(status_text)
    
    def _on_figure_leave(self, event):
        """
        Handle mouse leaving the figure.
        
        Args:
            event: Matplotlib figure leave event
        """
        self.clear_annotations()
        if self.status_var:
            self._clear_status_display()
        self.last_cursor_pos = None

    def update_annotations(self):
        """
        Update annotations based on last cursor position.
        Use this when curve data changes but cursor hasn't moved.
        """
        if self.last_cursor_pos is not None and self.show_points:
            # Create a mock event
            class MockEvent:
                def __init__(self, x, y, inaxes):
                    self.x = x
                    self.y = y
                    self.inaxes = inaxes
            
            mock_event = MockEvent(self.last_cursor_pos[0], self.last_cursor_pos[1], self.ax)
            self._on_mouse_move(mock_event)