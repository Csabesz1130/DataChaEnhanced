import numpy as np
from src.utils.logger import app_logger

class CurvePointTracker:
    """
    Enhanced point tracker that shows point information in the status bar.
    """
    
    def __init__(self, figure, ax, status_var=None):
        """
        Initialize the curve point tracker that always works when hovering over points.
        
        Args:
            figure: Matplotlib figure
            ax: Matplotlib axes
            status_var: StringVar for status display
        """
        self.fig = figure
        self.ax = ax
        self.status_var = status_var  # For status bar display
        self.annotations = {}
        self.curve_data = {
            'orange': {'data': None, 'times': None, 'visible': True},  # Always visible by default
            'blue': {'data': None, 'times': None, 'visible': True},
            'magenta': {'data': None, 'times': None, 'visible': True},
            'purple_hyperpol': {'data': None, 'times': None, 'visible': True},
            'purple_depol': {'data': None, 'times': None, 'visible': True}
        }
        
        # This flag only controls annotations, not basic point tracking
        self.show_annotations = False
        self.last_cursor_pos = None
        self.current_time = 0
        self.current_value = 0
        
        # Display names for the curves
        self.curve_names = {
            'orange': 'Orange',
            'blue': 'Blue',
            'magenta': 'Magenta',
            'purple_hyperpol': 'Purple Hyperpol',
            'purple_depol': 'Purple Depol'
        }
        
        # Color mapping for annotation text
        self.colors = {
            'orange': 'orange',
            'blue': 'blue',
            'magenta': 'magenta',
            'purple_hyperpol': 'purple',
            'purple_depol': 'darkviolet'
        }
        
        # Store slice information for mapping
        self._hyperpol_slice = (1028, 1227)  # From log data
        self._depol_slice = (828, 1028)      # From log data
        
        # Connect events immediately
        self._connect()
        
        app_logger.info("Point tracker initialized with always-on tracking")
        
    def _connect(self):
        """Connect to matplotlib event callbacks"""
        # Disconnect existing connections if they exist
        if hasattr(self, 'cid_move'):
            self.fig.canvas.mpl_disconnect(self.cid_move)
        if hasattr(self, 'cid_figure'):
            self.fig.canvas.mpl_disconnect(self.cid_figure)
            
        # Create new connections
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.cid_figure = self.fig.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)
        app_logger.debug("Point tracker event connections established")
    
    def _disconnect(self):
        """Disconnect from matplotlib event callbacks"""
        if hasattr(self, 'cid_move'):
            self.fig.canvas.mpl_disconnect(self.cid_move)
        if hasattr(self, 'cid_figure'):
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
            if data is None:
                app_logger.debug(f"Ignoring None data for {curve_type}")
                return
                
            self.curve_data[curve_type]['data'] = data
            self.curve_data[curve_type]['times'] = times
            self.curve_data[curve_type]['visible'] = visible
            app_logger.debug(f"Set {curve_type} curve data with {len(data)} points, visible={visible}")
    
    def set_show_points(self, show):
        """
        Set whether to show points.
        
        Args:
            show: Boolean flag to show or hide points
        """
        self.show_points = show
        app_logger.debug(f"Point tracking set to {show}")
        
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
    
    def _get_corresponding_orange_point(self, curve_type, point_idx):
        """
        Get the corresponding orange curve point index with improved logic.
        
        Args:
            curve_type: Type of curve
            point_idx: Index of the point on the curve
                
        Returns:
            Corresponding orange point index or None
        """
        # Return None if orange curve data is not available
        if self.curve_data['orange']['data'] is None or len(self.curve_data['orange']['data']) == 0:
            return None
            
        # Get the orange data length
        orange_len = len(self.curve_data['orange']['data'])
        
        # For normalized (blue) curve, the mapping is straightforward
        if curve_type == 'blue':
            if point_idx < orange_len:
                return point_idx
            else:
                app_logger.debug(f"Blue point {point_idx} out of orange range ({orange_len})")
                return None
        
        # For averaged (magenta) curve, calculate scaled index based on relative lengths
        elif curve_type == 'magenta':
            magenta_len = len(self.curve_data[curve_type]['data'])
            
            if magenta_len > 0:
                # Scale the index proportionally
                scaled_idx = int(point_idx * orange_len / magenta_len)
                if scaled_idx < orange_len:
                    return scaled_idx
                else:
                    app_logger.debug(f"Scaled magenta point {point_idx}->{scaled_idx} out of orange range ({orange_len})")
                    return None
        
        # For purple curves, try multiple approaches to find the corresponding orange point
        elif curve_type in ['purple_hyperpol', 'purple_depol']:
            # First try to get processor reference from window
            app = None
            try:
                if hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'window'):
                    app = self.fig.canvas.manager.window
            except:
                app = None
            
            processor = getattr(app, 'action_potential_processor', None)
            
            if processor is not None:
                # Try to get indices directly from processor
                if curve_type == 'purple_hyperpol':
                    # Try the different attributes where these indices might be stored
                    if hasattr(processor, 'hyperpol_indices') and processor.hyperpol_indices:
                        try:
                            if len(processor.hyperpol_indices) > point_idx:
                                return processor.hyperpol_indices[point_idx]
                            else:
                                start_idx = processor.hyperpol_indices[0]
                                return start_idx + point_idx
                        except (IndexError, AttributeError) as e:
                            app_logger.debug(f"Error getting hyperpol index: {e}")
                    
                    # Fall back to slice information
                    if hasattr(processor, '_hyperpol_slice'):
                        start_idx = processor._hyperpol_slice[0]
                        if start_idx + point_idx < orange_len:
                            return start_idx + point_idx
                
                elif curve_type == 'purple_depol':
                    # Try the different attributes where these indices might be stored
                    if hasattr(processor, 'depol_indices') and processor.depol_indices:
                        try:
                            if len(processor.depol_indices) > point_idx:
                                return processor.depol_indices[point_idx]
                            else:
                                start_idx = processor.depol_indices[0]
                                return start_idx + point_idx
                        except (IndexError, AttributeError) as e:
                            app_logger.debug(f"Error getting depol index: {e}")
                    
                    # Fall back to slice information
                    if hasattr(processor, '_depol_slice'):
                        start_idx = processor._depol_slice[0]
                        if start_idx + point_idx < orange_len:
                            return start_idx + point_idx
        
        # No corresponding orange point found
        return None
    
    def _get_nearest_point(self, x, y, curve_type):
        """
        High-performance version of the nearest point detector
        that's optimized for speed while maintaining precision.
        """
        # Get curve data
        curve_data = self.curve_data[curve_type]
        if not curve_data['visible'] or curve_data['data'] is None:
            return None
        
        data = curve_data['data']
        times = curve_data['times']
        
        if len(data) == 0 or times is None or len(times) == 0:
            return None
        
        # Convert mouse coordinates to data coordinates
        data_x, data_y = self.ax.transData.inverted().transform((x, y))
        
        # Handle time units (convert ms to seconds if needed)
        if self.ax.get_xlabel().lower() in ('time (ms)', 'time(ms)'):
            compare_x = data_x / 1000.0  # Convert ms to s
        else:
            compare_x = data_x  # Already in same units
        
        # Use fast vectorized operations to find closest point
        distances = np.abs(times - compare_x)
        idx = np.argmin(distances)
        
        # Use adaptive thresholds based on data range
        x_range = np.ptp(times) * 0.05  # 5% of time range
        y_range = np.ptp(data) * 0.20   # 20% of data range
        
        # Strict x-threshold, generous y-threshold for better usability
        x_threshold = min(x_range, 0.005)  # Cap at 5ms for precision
        y_threshold = max(y_range, 50.0)   # At least 50pA for usability
        
        # Calculate distance to the closest point
        x_distance = abs(times[idx] - compare_x)
        y_distance = abs(data[idx] - data_y)
        
        # Return point if within thresholds
        if x_distance <= x_threshold and y_distance <= y_threshold:
            return (idx, np.hypot(x_distance, y_distance), times[idx], data[idx])
        
        return None
    
    def force_update(self, enable=True):
        """
        Force-enable point tracking and update all curve data.
        
        Args:
            enable: Whether to enable point tracking
        """
        app_logger.info(f"Force-updating point tracker (enable={enable})")
        
        try:
            # First, ensure we're connected to events
            self._connect()
            
            # Enable point tracking
            self.set_show_points(enable)
            
            # Get app reference for data access
            app = None
            try:
                if hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'window'):
                    app = self.fig.canvas.manager.window
            except:
                app = None
            
            # Update curve data from processor if available
            if app and hasattr(app, 'action_potential_processor') and app.action_potential_processor is not None:
                processor = app.action_potential_processor
                app_logger.debug("Retrieved processor from app for force_update")
                
                # Try to update curve data for tracking
                try:
                    # Update orange curve
                    if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
                        self.set_curve_data(
                            'orange', 
                            processor.orange_curve, 
                            getattr(processor, 'orange_curve_times', None),
                            True
                        )
                        app_logger.debug(f"Updated orange curve: {len(processor.orange_curve)} points")
                    
                    # Update blue curve
                    if hasattr(processor, 'normalized_curve') and processor.normalized_curve is not None:
                        self.set_curve_data(
                            'blue',
                            processor.normalized_curve,
                            getattr(processor, 'normalized_curve_times', None),
                            True
                        )
                        app_logger.debug(f"Updated blue curve: {len(processor.normalized_curve)} points")
                    
                    # Update magenta curve
                    if hasattr(processor, 'average_curve') and processor.average_curve is not None:
                        self.set_curve_data(
                            'magenta',
                            processor.average_curve,
                            getattr(processor, 'average_curve_times', None),
                            True
                        )
                        app_logger.debug(f"Updated magenta curve: {len(processor.average_curve)} points")
                    
                    # Update purple curves
                    if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
                        self.set_curve_data(
                            'purple_hyperpol',
                            processor.modified_hyperpol,
                            getattr(processor, 'modified_hyperpol_times', None),
                            True
                        )
                        app_logger.debug(f"Updated purple hyperpol: {len(processor.modified_hyperpol)} points")
                    
                    if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
                        self.set_curve_data(
                            'purple_depol',
                            processor.modified_depol,
                            getattr(processor, 'modified_depol_times', None),
                            True
                        )
                        app_logger.debug(f"Updated purple depol: {len(processor.modified_depol)} points")
                    
                    # Store processor reference
                    self.processor = processor
                    
                    return True
                    
                except Exception as e:
                    app_logger.error(f"Error updating curve data: {str(e)}")
                    return False
            else:
                app_logger.warning("No processor available for force_update")
                return False
                
        except Exception as e:
            app_logger.error(f"Force update failed: {str(e)}")
            return False
    
    def _connect(self):
        """Connect to matplotlib event callbacks with improved error handling"""
        try:
            # Disconnect existing connections if they exist
            if hasattr(self, 'cid_move'):
                try:
                    self.fig.canvas.mpl_disconnect(self.cid_move)
                except:
                    pass
                
            if hasattr(self, 'cid_figure'):
                try:
                    self.fig.canvas.mpl_disconnect(self.cid_figure)
                except:
                    pass
                
            # Create new connections
            self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.cid_figure = self.fig.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)
            
            # Force a canvas draw to ensure connections are active
            self.fig.canvas.draw_idle()
            
            app_logger.info("Point tracker event connections established")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to connect point tracker events: {str(e)}")
            return False

    def _on_mouse_move(self, event):
        """
        Handle mouse movement events - always report point information
        even if show_annotations is disabled.
        """
        # Quick exit if not over axes
        if not event.inaxes or event.inaxes != self.ax:
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set("")
            self.clear_annotations()
            return
        
        # Store basic position info
        self.last_cursor_pos = (event.x, event.y)
        self.current_time = event.xdata  # In ms
        self.current_value = event.ydata  # In pA
        basic_info = f"Time: {self.current_time:.1f} ms, Current: {self.current_value:.1f} pA"
        
        # Always check for nearby points regardless of show_annotations flag
        found_point = False
        status_parts = [basic_info]
        
        # Check curves in priority order for better performance
        priority_curves = ['orange', 'purple_hyperpol', 'purple_depol', 'magenta', 'blue']
        
        for curve_type in priority_curves:
            # Skip if curve data isn't available
            if (self.curve_data[curve_type]['data'] is None or 
                len(self.curve_data[curve_type]['data']) == 0):
                continue
                
            # Try to find nearest point
            point_info = self._get_nearest_point(event.x, event.y, curve_type)
            if point_info is not None:
                idx, distance, x_val, y_val = point_info
                
                # Get corresponding orange point
                orange_idx = self._get_corresponding_orange_point(curve_type, idx)
                
                # Format for status bar (use 1-based indexing for display)
                point_text = f"{self.curve_names[curve_type]} Point: {idx+1}"
                if orange_idx is not None and curve_type != 'orange':
                    point_text += f" (Orange: {orange_idx+1})"
                
                status_parts.append(point_text)
                found_point = True
                
                # Only show annotations if that feature is enabled
                if self.show_annotations:
                    self._add_annotation(curve_type, idx, x_val, y_val, orange_idx)
                
                # For better performance, only show the first point found
                break
        
        # Always update status bar with point info, regardless of annotation settings
        if hasattr(self, 'status_var') and self.status_var:
            if found_point:
                self.status_var.set(" | ".join(status_parts))
            else:
                self.status_var.set(basic_info)
                # Clear annotations when no point is found
                self.clear_annotations()

    def _update_annotations_from_points(self, nearby_points):
        """
        Update plot annotations using pre-calculated nearby points.
        
        Args:
            nearby_points: List of dictionaries with point information
        """
        # Process each curve type
        for curve_type in self.curve_data:
            # Find the point for this curve type
            point_info = next((p for p in nearby_points if p['curve_type'] == curve_type), None)
            
            if point_info is not None:
                # Extract relevant information
                idx = point_info['index']
                x_val = point_info['x_val']
                y_val = point_info['y_val']
                orange_idx = point_info['orange_idx']
                
                # Create annotation text
                color_name = curve_type.split('_')[0].capitalize()
                if '_' in curve_type:
                    color_name += f" {curve_type.split('_')[1].capitalize()}"
                
                if orange_idx is not None:
                    text = f"{color_name} point: {idx} [Orange: {orange_idx}]"
                else:
                    text = f"{color_name} point: {idx}"
                
                # Add or update annotation
                if curve_type in self.annotations and self.annotations[curve_type] is not None:
                    self.annotations[curve_type].set_text(text)
                    self.annotations[curve_type].set_position((x_val * 1000, y_val))  # Convert s to ms if needed
                else:
                    offset_x, offset_y = self.offsets[curve_type]
                    self.annotations[curve_type] = self.ax.annotate(
                        text, xy=(x_val * 1000, y_val),  # Convert s to ms if needed
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        color=self.colors[curve_type],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color=self.colors[curve_type])
                    )
            elif curve_type in self.annotations and self.annotations[curve_type] is not None:
                # Remove annotation if no point found for this curve
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