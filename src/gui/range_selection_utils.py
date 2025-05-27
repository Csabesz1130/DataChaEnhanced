"""
Range selection utilities for the Signal Analyzer application.
This module provides enhanced range selection, feedback, and integration
functionality for the ActionPotentialTab.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from src.utils.logger import app_logger

class RangeSelectionManager:
    def __init__(self, parent, container_frame, callback):
        """
        Initialize the range selection manager.
        
        Args:
            parent: The parent widget (ActionPotentialTab instance)
            container_frame: The frame to add controls to
            callback: Function to call when ranges change
        """
        self.parent = parent
        self.frame = container_frame
        self.update_callback = callback
        
        # For tooltip display
        self.tooltip_label = None
        
        # Variables for range selection
        self.hyperpol_start = tk.IntVar(value=0)
        self.hyperpol_end = tk.IntVar(value=200)
        self.depol_start = tk.IntVar(value=0)
        self.depol_end = tk.IntVar(value=200)
        
        # Store entries and sliders for later access
        self.slider_controls = {}
        self.entry_fields = {}
        
        # Setup the UI components
        self.setup_range_controls()
        
    def setup_range_controls(self):
        """Setup sliders for hyperpolarization and depolarization integration ranges with enhanced feedback."""
        range_frame = ttk.LabelFrame(self.frame, text="Integration Ranges")
        range_frame.pack(fill='x', padx=5, pady=5)

        # Setup floating display for real-time feedback
        self.setup_floating_display(range_frame)

        # Hyperpolarization range
        hyperpol_frame = ttk.LabelFrame(range_frame, text="Hyperpolarization")
        hyperpol_frame.pack(fill='x', padx=5, pady=5)

        # Start slider for hyperpol
        hyperpol_start_frame = ttk.Frame(hyperpol_frame)
        hyperpol_start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_start_frame, text="Start:").pack(side='left')
        self.hyperpol_start_display = ttk.Label(hyperpol_start_frame, width=5)
        self.hyperpol_start_display.pack(side='right')
        
        self.hyperpol_start_slider = ttk.Scale(
            hyperpol_frame, 
            from_=0, 
            to=199,
            variable=self.hyperpol_start,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.hyperpol_start_display, 'hyperpol_start')
        )
        self.hyperpol_start_slider.pack(fill='x', padx=5)
        self.hyperpol_start_slider.configure(state='disabled')  # Initially disabled
        self.slider_controls['hyperpol_start'] = self.hyperpol_start_slider

        # End slider for hyperpol
        hyperpol_end_frame = ttk.Frame(hyperpol_frame)
        hyperpol_end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_end_frame, text="End:").pack(side='left')
        self.hyperpol_end_display = ttk.Label(hyperpol_end_frame, width=5)
        self.hyperpol_end_display.pack(side='right')
        
        self.hyperpol_end_slider = ttk.Scale(
            hyperpol_frame, 
            from_=1, 
            to=200,
            variable=self.hyperpol_end,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.hyperpol_end_display, 'hyperpol_end')
        )
        self.hyperpol_end_slider.pack(fill='x', padx=5)
        self.hyperpol_end_slider.configure(state='disabled')  # Initially disabled
        self.slider_controls['hyperpol_end'] = self.hyperpol_end_slider

        # Depolarization range
        depol_frame = ttk.LabelFrame(range_frame, text="Depolarization")
        depol_frame.pack(fill='x', padx=5, pady=5)

        # Start slider for depol
        depol_start_frame = ttk.Frame(depol_frame)
        depol_start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_start_frame, text="Start:").pack(side='left')
        self.depol_start_display = ttk.Label(depol_start_frame, width=5)
        self.depol_start_display.pack(side='right')
        
        self.depol_start_slider = ttk.Scale(
            depol_frame, 
            from_=0, 
            to=199,
            variable=self.depol_start,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.depol_start_display, 'depol_start')
        )
        self.depol_start_slider.pack(fill='x', padx=5)
        self.depol_start_slider.configure(state='disabled')  # Initially disabled
        self.slider_controls['depol_start'] = self.depol_start_slider

        # End slider for depol
        depol_end_frame = ttk.Frame(depol_frame)
        depol_end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_end_frame, text="End:").pack(side='left')
        self.depol_end_display = ttk.Label(depol_end_frame, width=5)
        self.depol_end_display.pack(side='right')
        
        self.depol_end_slider = ttk.Scale(
            depol_frame, 
            from_=1, 
            to=200,
            variable=self.depol_end,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.depol_end_display, 'depol_end')
        )
        self.depol_end_slider.pack(fill='x', padx=5)
        self.depol_end_slider.configure(state='disabled')  # Initially disabled
        self.slider_controls['depol_end'] = self.depol_end_slider

        # Setup keyboard navigation and direct entry fields
        self.setup_keyboard_navigation(hyperpol_frame, depol_frame)

        # Help text
        ttk.Label(
            range_frame,
            text="Adjust sliders to set integration range for each curve (0.5ms per point)\nSliders snap to a grid of 5",
            wraplength=300,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=5)
        
        # Initialize display labels
        self.update_slider_display(self.hyperpol_start.get(), self.hyperpol_start_display, 'hyperpol_start')
        self.update_slider_display(self.hyperpol_end.get(), self.hyperpol_end_display, 'hyperpol_end')
        self.update_slider_display(self.depol_start.get(), self.depol_start_display, 'depol_start')
        self.update_slider_display(self.depol_end.get(), self.depol_end_display, 'depol_end')

    def setup_floating_display(self, parent_frame):
        """Create a floating display for real-time range information."""
        self.floating_display_frame = ttk.LabelFrame(
            parent_frame, 
            text="Range Information",
            padding=5
        )
        self.floating_display_frame.pack(fill='x', padx=5, pady=5)
        
        # Range values display
        self.range_display = ttk.Label(
            self.floating_display_frame,
            text="Hyperpol: 0-200\nDepol: 0-200",
            justify="left"
        )
        self.range_display.pack(fill='x', padx=2, pady=2)
        
        # Time values display
        self.time_display = ttk.Label(
            self.floating_display_frame,
            text="Hyperpol: 0.0-100.0 ms\nDepol: 0.0-100.0 ms",
            justify="left"
        )
        self.time_display.pack(fill='x', padx=2, pady=2)
        
        # Integral values display
        self.integral_display = ttk.Label(
            self.floating_display_frame,
            text="Hyperpol integral: 0.00 pC\nDepol integral: 0.00 pC",
            justify="left"
        )
        self.integral_display.pack(fill='x', padx=2, pady=2)

    def update_slider_display(self, value, label, slider_type):
        """Update display label for a slider with improved coordination between ranges."""
        try:
            # Get raw value
            raw_val = int(float(value))
            
            # Check if we should snap to grid
            apply_snap = self.should_apply_snap_to_grid()
            
            # Apply snapping only if in the right mode
            if apply_snap:
                val = round(raw_val / 5) * 5
            else:
                val = raw_val
            
            # Update the label if provided
            if label is not None:
                label.config(text=str(val))
            
            # Define min gap based on mode
            min_gap = 5 if apply_snap else 1
            
            # Handle constraints differently based on slider type
            if slider_type == 'hyperpol_start':
                # Ensure hyperpol_start < hyperpol_end
                if val >= self.hyperpol_end.get() - min_gap:
                    # Adjust end to maintain gap
                    new_end = val + min_gap
                    self.hyperpol_end.set(new_end)
                    
            elif slider_type == 'hyperpol_end':
                # Ensure hyperpol_end > hyperpol_start
                if val <= self.hyperpol_start.get() + min_gap:
                    # Adjust start to maintain gap
                    new_start = val - min_gap
                    if new_start >= 0:  # Don't go below 0
                        self.hyperpol_start.set(new_start)
                    else:
                        # If we can't move start below 0, adjust end instead
                        self.hyperpol_end.set(min_gap)
                
                # REMOVE THIS CONSTRAINT - Don't force depol to move when hyperpol changes
                # if val >= self.depol_start.get() - min_gap:
                #     new_depol_start = val + min_gap
                #     self.depol_start.set(new_depol_start)
                #     if new_depol_start >= self.depol_end.get() - min_gap:
                #         new_depol_end = new_depol_start + min_gap
                #         self.depol_end.set(new_depol_end)
                    
            elif slider_type == 'depol_start':
                # Ensure depol_start < depol_end
                if val >= self.depol_end.get() - min_gap:
                    # Adjust end to maintain gap
                    new_end = val + min_gap
                    self.depol_end.set(new_end)
                
                # REMOVE THIS CONSTRAINT - Don't force hyperpol to move when depol changes
                # if val <= self.hyperpol_end.get() + min_gap:
                #     new_hyperpol_end = val - min_gap
                #     if new_hyperpol_end >= self.hyperpol_start.get() + min_gap:
                #         self.hyperpol_end.set(new_hyperpol_end)
                #     else:
                #         new_val = self.hyperpol_end.get() + min_gap
                #         self.depol_start.set(new_val)
                #         val = new_val
                    
            elif slider_type == 'depol_end':
                # Ensure depol_end > depol_start
                if val <= self.depol_start.get() + min_gap:
                    # Adjust start to maintain gap
                    new_start = val - min_gap
                    self.depol_start.set(new_start)
                
            # Update entry fields if available
            if slider_type in self.entry_fields:
                entry = self.entry_fields[slider_type]
                entry.delete(0, tk.END)
                entry.insert(0, str(val))
            
            # Update range display and calculate integrals
            self.update_range_display()
            self.calculate_range_integral()
            
            # Trigger the callback
            self.trigger_update_callback()
            
        except Exception as e:
            app_logger.error(f"Error updating slider display: {str(e)}")

    def show_tooltip(self, message, duration=1500):
        """Show a temporary tooltip message."""
        if hasattr(self, 'tooltip_label') and self.tooltip_label is not None:
            self.tooltip_label.destroy()
        
        self.tooltip_label = ttk.Label(
            self.frame,
            text=message,
            background="#FFFFCC",
            relief="solid",
            borderwidth=1,
            font=('TkDefaultFont', 9)
        )
        self.tooltip_label.place(
            relx=0.5, 
            rely=0.2, 
            anchor="center"
        )
        
        # Auto-destroy after specified duration
        self.tooltip_label.after(duration, lambda: self.tooltip_label.destroy() if hasattr(self, 'tooltip_label') and self.tooltip_label is not None else None)

    def update_range_display(self):
        """Update the display of current integration ranges with time values."""
        if not hasattr(self, 'range_display') or not hasattr(self, 'time_display'):
            return
            
        # Get current ranges
        hyperpol_start = self.hyperpol_start.get()
        hyperpol_end = self.hyperpol_end.get()
        depol_start = self.depol_start.get()
        depol_end = self.depol_end.get()
        
        # Update range display
        self.range_display.config(
            text=f"Hyperpol: {hyperpol_start}-{hyperpol_end}\n"
                f"Depol: {depol_start}-{depol_end}"
        )
        
        # Calculate time values (assuming 0.5ms per point)
        hyperpol_start_time = hyperpol_start * 0.5
        hyperpol_end_time = hyperpol_end * 0.5
        depol_start_time = depol_start * 0.5
        depol_end_time = depol_end * 0.5
        
        # Update time display
        self.time_display.config(
            text=f"Hyperpol: {hyperpol_start_time:.1f}-{hyperpol_end_time:.1f} ms\n"
                f"Depol: {depol_start_time:.1f}-{depol_end_time:.1f} ms"
        )

    def calculate_range_integral(self):
        """
        Calculate and display the integral for the current ranges.
        Uses the local processor reference if available.
        """
        if not hasattr(self, 'integral_display'):
            return

        # First try getting processor from direct reference
        processor = getattr(self, 'processor', None)
        
        # If not available, try getting from parent class
        if processor is None and hasattr(self, 'parent'):
            processor = getattr(self.parent, 'processor', None)
            
        # If still not available, try getting from app through parent
        if processor is None and hasattr(self, 'parent') and hasattr(self.parent, 'parent'):
            app = self.parent.parent.master
            processor = getattr(app, 'action_potential_processor', None)
        
        # Check if we have a valid processor with necessary data
        if (processor is None or 
            not hasattr(processor, 'modified_hyperpol') or 
            not hasattr(processor, 'modified_depol') or
            processor.modified_hyperpol is None or
            processor.modified_depol is None):
            
            self.integral_display.config(text="No purple curves available")
            return
        
        # Get current ranges
        ranges = self.get_integration_ranges()
        
        try:
            # Calculate integrals for both ranges
            hyperpol_range = ranges['hyperpol']
            hyperpol_start = hyperpol_range['start']
            hyperpol_end = hyperpol_range['end']
            
            depol_range = ranges['depol']
            depol_start = depol_range['start']
            depol_end = depol_range['end']
            
            # Check if indices are valid
            if (hyperpol_start < len(processor.modified_hyperpol) and 
                hyperpol_end <= len(processor.modified_hyperpol) and
                depol_start < len(processor.modified_depol) and
                depol_end <= len(processor.modified_depol)):
                
                # Calculate hyperpol integral
                hyperpol_data = processor.modified_hyperpol[hyperpol_start:hyperpol_end]
                hyperpol_times = processor.modified_hyperpol_times[hyperpol_start:hyperpol_end]
                if len(hyperpol_data) > 1:
                    hyperpol_integral = np.trapz(hyperpol_data, x=hyperpol_times * 1000)
                else:
                    hyperpol_integral = 0
                
                # Calculate depol integral
                depol_data = processor.modified_depol[depol_start:depol_end]
                depol_times = processor.modified_depol_times[depol_start:depol_end]
                if len(depol_data) > 1:
                    depol_integral = np.trapz(depol_data, x=depol_times * 1000)
                else:
                    depol_integral = 0
                
                # Calculate the capacitance if available
                capacitance_text = ""
                if hasattr(processor, 'params'):
                    voltage_diff = abs(processor.params.get('V2', 0) - processor.params.get('V0', -80))
                    if voltage_diff > 0:
                        capacitance = abs(hyperpol_integral - depol_integral) / voltage_diff
                        capacitance_text = f"\nCapacitance: {capacitance:.3f} nF"
                
                # Update display with more precision and formatting
                self.integral_display.config(
                    text=(f"Hyperpol integral: {hyperpol_integral:.3f} pC\n"
                        f"Depol integral: {depol_integral:.3f} pC"
                        f"{capacitance_text}")
                )
                
                # Store results for later access
                self.current_integrals = {
                    'hyperpol_integral': hyperpol_integral,
                    'depol_integral': depol_integral,
                    'voltage_diff': voltage_diff if 'voltage_diff' in locals() else 0,
                    'capacitance': capacitance if 'capacitance' in locals() else 0
                }
                
                return self.current_integrals
            else:
                self.integral_display.config(text="Range indices out of bounds")
        
        except Exception as e:
            app_logger.error(f"Error calculating range integrals: {str(e)}")
            self.integral_display.config(text=f"Error calculating integrals")
            return None

    def setup_keyboard_navigation(self, hyperpol_frame, depol_frame):
        """Set up keyboard navigation and direct entry fields for precise slider adjustments."""
        # Hyperpol start direct entry
        hyperpol_start_entry_frame = ttk.Frame(hyperpol_frame)
        hyperpol_start_entry_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_start_entry_frame, text="Direct input:").pack(side='left')
        self.hyperpol_start_entry = ttk.Entry(hyperpol_start_entry_frame, width=5)
        self.hyperpol_start_entry.pack(side='left', padx=5)
        self.hyperpol_start_entry.insert(0, str(self.hyperpol_start.get()))
        ttk.Button(
            hyperpol_start_entry_frame, 
            text="Set", 
            command=lambda: self.set_from_entry('hyperpol_start')
        ).pack(side='left')
        self.entry_fields['hyperpol_start'] = self.hyperpol_start_entry
        
        # Hyperpol end direct entry
        hyperpol_end_entry_frame = ttk.Frame(hyperpol_frame)
        hyperpol_end_entry_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_end_entry_frame, text="Direct input:").pack(side='left')
        self.hyperpol_end_entry = ttk.Entry(hyperpol_end_entry_frame, width=5)
        self.hyperpol_end_entry.pack(side='left', padx=5)
        self.hyperpol_end_entry.insert(0, str(self.hyperpol_end.get()))
        ttk.Button(
            hyperpol_end_entry_frame, 
            text="Set", 
            command=lambda: self.set_from_entry('hyperpol_end')
        ).pack(side='left')
        self.entry_fields['hyperpol_end'] = self.hyperpol_end_entry
        
        # Depol start direct entry
        depol_start_entry_frame = ttk.Frame(depol_frame)
        depol_start_entry_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_start_entry_frame, text="Direct input:").pack(side='left')
        self.depol_start_entry = ttk.Entry(depol_start_entry_frame, width=5)
        self.depol_start_entry.pack(side='left', padx=5)
        self.depol_start_entry.insert(0, str(self.depol_start.get()))
        ttk.Button(
            depol_start_entry_frame, 
            text="Set", 
            command=lambda: self.set_from_entry('depol_start')
        ).pack(side='left')
        self.entry_fields['depol_start'] = self.depol_start_entry
        
        # Depol end direct entry
        depol_end_entry_frame = ttk.Frame(depol_frame)
        depol_end_entry_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_end_entry_frame, text="Direct input:").pack(side='left')
        self.depol_end_entry = ttk.Entry(depol_end_entry_frame, width=5)
        self.depol_end_entry.pack(side='left', padx=5)
        self.depol_end_entry.insert(0, str(self.depol_end.get()))
        ttk.Button(
            depol_end_entry_frame, 
            text="Set", 
            command=lambda: self.set_from_entry('depol_end')
        ).pack(side='left')
        self.entry_fields['depol_end'] = self.depol_end_entry
        
        # Bind keyboard events to sliders
        self.hyperpol_start_slider.bind("<Left>", lambda e: self.adjust_slider('hyperpol_start', -5))
        self.hyperpol_start_slider.bind("<Right>", lambda e: self.adjust_slider('hyperpol_start', 5))
        self.hyperpol_end_slider.bind("<Left>", lambda e: self.adjust_slider('hyperpol_end', -5))
        self.hyperpol_end_slider.bind("<Right>", lambda e: self.adjust_slider('hyperpol_end', 5))
        
        self.depol_start_slider.bind("<Left>", lambda e: self.adjust_slider('depol_start', -5))
        self.depol_start_slider.bind("<Right>", lambda e: self.adjust_slider('depol_start', 5))
        self.depol_end_slider.bind("<Left>", lambda e: self.adjust_slider('depol_end', -5))
        self.depol_end_slider.bind("<Right>", lambda e: self.adjust_slider('depol_end', 5))

    def adjust_slider(self, slider_name, amount):
        """Adjust a slider by the specified amount using keyboard."""
        slider_var = getattr(self, slider_name)
        current_value = slider_var.get()
        new_value = current_value + amount
        
        # Ensure the new value is within slider bounds
        slider = self.slider_controls.get(slider_name)
        if slider:
            min_val = int(float(slider.cget('from')))
            max_val = int(float(slider.cget('to')))
            
            if min_val <= new_value <= max_val:
                slider_var.set(new_value)
                # Update the corresponding entry field
                entry = self.entry_fields.get(slider_name)
                if entry:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(new_value))
                
                # Get the corresponding display label
                label_name = f"{slider_name}_display"
                if hasattr(self, label_name):
                    label = getattr(self, label_name)
                    # Update display
                    self.update_slider_display(new_value, label, slider_name)

    def set_from_entry(self, slider_name):
        """Set slider value from entry field with validation."""
        entry = self.entry_fields.get(slider_name)
        if not entry:
            return
            
        try:
            value = int(entry.get())
            
            # Get slider bounds
            slider = self.slider_controls.get(slider_name)
            if slider:
                min_val = int(float(slider.cget('from')))
                max_val = int(float(slider.cget('to')))
                
                # Validate and apply the value
                if min_val <= value <= max_val:
                    # Apply snap-to-grid
                    snapped_value = round(value / 5) * 5
                    if value != snapped_value:
                        self.show_tooltip(f"Snapped to grid: {snapped_value}")
                        entry.delete(0, tk.END)
                        entry.insert(0, str(snapped_value))
                        
                    # Set the slider value
                    slider_var = getattr(self, slider_name)
                    slider_var.set(snapped_value)
                    
                    # Get the corresponding display label
                    label_name = f"{slider_name}_display"
                    if hasattr(self, label_name):
                        label = getattr(self, label_name)
                        # Update display
                        self.update_slider_display(snapped_value, label, slider_name)
                else:
                    self.show_tooltip(f"Value must be between {min_val} and {max_val}")
                    # Reset entry to current slider value
                    entry.delete(0, tk.END)
                    entry.insert(0, str(getattr(self, slider_name).get()))
        except ValueError:
            self.show_tooltip("Please enter a valid number")
            # Reset entry to current slider value
            entry.delete(0, tk.END)
            entry.insert(0, str(getattr(self, slider_name).get()))

    def get_integration_ranges(self):
        """Get current integration ranges for both curves."""
        return {
            'hyperpol': {
                'start': self.hyperpol_start.get(),
                'end': self.hyperpol_end.get()
            },
            'depol': {
                'start': self.depol_start.get(),
                'end': self.depol_end.get()
            }
        }

    def set_integration_ranges(self, ranges):
        """Set integration ranges from external source."""
        try:
            if 'hyperpol' in ranges:
                hyperpol = ranges['hyperpol']
                if 'start' in hyperpol:
                    self.hyperpol_start.set(hyperpol['start'])
                if 'end' in hyperpol:
                    self.hyperpol_end.set(hyperpol['end'])
                    
            if 'depol' in ranges:
                depol = ranges['depol']
                if 'start' in depol:
                    self.depol_start.set(depol['start'])
                if 'end' in depol:
                    self.depol_end.set(depol['end'])
                    
            # Update displays
            self.update_range_display()
            self.calculate_range_integral()
            
            # Update entry fields
            for name, entry in self.entry_fields.items():
                var = getattr(self, name)
                entry.delete(0, tk.END)
                entry.insert(0, str(var.get()))
                
        except Exception as e:
            app_logger.error(f"Error setting integration ranges: {str(e)}")

    def trigger_update_callback(self):
        """Trigger the callback with current ranges."""
        if self.update_callback:
            self.update_callback({
                'integration_ranges': self.get_integration_ranges(),
                'show_points': True if hasattr(self.parent, 'show_points') and self.parent.show_points.get() else False
            })

    def add_range_visualizations(self, ax, processor):
        """Add range visualizations to the given matplotlib axis with enhanced visibility."""
        if processor is None or not hasattr(processor, 'modified_hyperpol_times'):
            return False
                
        try:
            # Get current ranges
            ranges = self.get_integration_ranges()
            
            # Store marker references for later updates
            if not hasattr(self, 'range_markers'):
                self.range_markers = {
                    'hyperpol': {'start': None, 'end': None, 'span': None},
                    'depol': {'start': None, 'end': None, 'span': None}
                }
            
            # Remove any existing markers
            for range_type in ['hyperpol', 'depol']:
                for key in ['start', 'end', 'span']:
                    if self.range_markers[range_type][key] is not None:
                        try:
                            self.range_markers[range_type][key].remove()
                        except:
                            pass
                        self.range_markers[range_type][key] = None
            
            # Define arrow properties
            arrow_props = {
                'hyperpol': {
                    'start': dict(facecolor='blue', edgecolor='darkblue', width=3, headwidth=10, shrink=0.05),
                    'end': dict(facecolor='darkblue', edgecolor='blue', width=3, headwidth=10, shrink=0.05)
                },
                'depol': {
                    'start': dict(facecolor='red', edgecolor='darkred', width=3, headwidth=10, shrink=0.05),
                    'end': dict(facecolor='darkred', edgecolor='red', width=3, headwidth=10, shrink=0.05)
                }
            }
            
            # Get y-axis limits for arrow placement
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            
            # Process both range types separately
            for range_type in ['hyperpol', 'depol']:
                # Skip if range not provided
                if range_type not in ranges:
                    continue
                    
                # Get correct attributes based on range type
                if range_type == 'hyperpol':
                    times_attr = 'modified_hyperpol_times'
                    color = 'blue'
                    alpha = 0.2
                    zorder = 5
                else:  # depol
                    times_attr = 'modified_depol_times'
                    color = '#ff5555'  # Brighter red
                    alpha = 0.25
                    zorder = 6  # Higher than hyperpol for visibility
                    
                # Skip if times attribute doesn't exist or is empty
                if not hasattr(processor, times_attr) or getattr(processor, times_attr) is None:
                    continue
                    
                times = getattr(processor, times_attr)
                if len(times) == 0:
                    continue
                    
                # Extract range settings
                range_data = ranges[range_type]
                start_idx = range_data['start']
                end_idx = range_data['end']
                
                # Make sure indices are valid
                start_idx = min(max(0, start_idx), len(times)-1)
                end_idx = min(max(start_idx + 1, end_idx), len(times))
                
                # Get time values in milliseconds
                start_ms = times[start_idx] * 1000
                end_ms = times[min(end_idx-1, len(times)-1)] * 1000
                
                # Add shaded region with increased visibility
                self.range_markers[range_type]['span'] = ax.axvspan(
                    start_ms, end_ms,
                    color=color, 
                    alpha=alpha,
                    label=f'{range_type.capitalize()} Range',
                    zorder=zorder
                )
                
                # Add arrows at boundaries for additional visibility
                arrow_y = ymax - y_range * 0.05
                
                self.range_markers[range_type]['start'] = ax.annotate(
                    '', xy=(start_ms, arrow_y), xytext=(start_ms, arrow_y + y_range*0.1),
                    arrowprops=arrow_props[range_type]['start'],
                    annotation_clip=False,
                    zorder=10
                )
                
                self.range_markers[range_type]['end'] = ax.annotate(
                    '', xy=(end_ms, arrow_y), xytext=(end_ms, arrow_y + y_range*0.1),
                    arrowprops=arrow_props[range_type]['end'],
                    annotation_clip=False,
                    zorder=10
                )
                
                app_logger.debug(f"Added {range_type} range visualization at {start_ms:.1f}-{end_ms:.1f}ms")
            
            return True
                
        except Exception as e:
            app_logger.error(f"Error adding range visualizations: {str(e)}")
            return False

    def enable_controls(self, enable=True):
        """Enable or disable all range selection controls."""
        state = 'normal' if enable else 'disabled'
        for slider in self.slider_controls.values():
            slider.configure(state=state)
        for entry in self.entry_fields.values():
            entry.configure(state=state)

    def get_current_integrals(self):
        """Get the most recently calculated integral values."""
        if hasattr(self, 'current_integrals'):
            return self.current_integrals
        return {
            'hyperpol_integral': 0.0,
            'depol_integral': 0.0
        }

    def should_apply_snap_to_grid(self):
        """Check if snapping to grid should be applied."""
        return True  # Mindig alkalmazzuk a rácshoz igazítást

    # Add these methods to the RangeSelectionManager class in range_selection_utils.py

def create_custom_slider_with_handle(self, parent_frame, label_text, variable, min_val, max_val, slider_type, color):
    """Create a custom slider with a handle and color coding."""
    # Create container frame for the slider and handle
    container = ttk.Frame(parent_frame)
    container.pack(fill='x', padx=5, pady=5)
    
    # Add label and value display
    label_frame = ttk.Frame(container)
    label_frame.pack(fill='x', pady=2)
    
    ttk.Label(label_frame, text=label_text).pack(side='left')
    display_label = ttk.Label(label_frame, width=5, background=color)
    display_label.pack(side='right')
    
    # Create slider frame with handle indicator
    slider_frame = ttk.Frame(container)
    slider_frame.pack(fill='x', pady=2)
    
    # Create the actual slider
    slider = ttk.Scale(
        slider_frame, 
        from_=min_val, 
        to=max_val,
        variable=variable,
        orient='horizontal',
        command=lambda v: self.update_slider_display(v, display_label, slider_type)
    )
    slider.pack(fill='x', side='left', expand=True)
    
    # Create handle button that follows the slider
    handle_image = self.create_handle_image(color)
    handle_button = ttk.Button(
        slider_frame, 
        image=handle_image,
        style=f"{color}.TButton",
        width=3
    )
    handle_button.image = handle_image  # Keep reference to prevent garbage collection
    handle_button.pack(side='left', padx=0)
    
    # Bind the button to move the slider
    handle_button.bind("<Button-1>", lambda e: self.start_drag(slider, slider_type))
    handle_button.bind("<B1-Motion>", lambda e: self.drag_slider(e, slider, slider_type))
    
    # Configure slider style based on color
    if hasattr(self, 'style'):
        slider.configure(style=f"{color}.Horizontal.TScale")
    
    # Store reference to the handle button for positioning updates
    setattr(self, f"{slider_type}_handle", handle_button)
    
    # Update display label
    self.update_slider_display(variable.get(), display_label, slider_type)
    
    return slider, display_label

def create_handle_image(self, color):
    """Create a colored handle image for the slider."""
    # Create a PhotoImage for the handle
    handle = tk.PhotoImage(width=15, height=20)
    
    # Map color names to RGB
    color_map = {
        'blue': '#4287f5',
        'darkblue': '#1c56c5',
        'red': '#f54242',
        'darkred': '#c51c1c'
    }
    
    # Get actual color code
    fill_color = color_map.get(color, color)
    
    # Draw a simple handle shape (arrow pointing down)
    for x in range(15):
        for y in range(20):
            # Draw triangle/arrow shape
            if (x >= 7-y//2 and x <= 7+y//2 and y < 10) or (3 <= x <= 11 and 10 <= y < 20):
                handle.put(fill_color, (x, y))
    
    return handle

def setup_styled_sliders(self):
    """Set up custom styles for sliders."""
    self.style = ttk.Style()
    
    # Create blue hyperpolarization slider style
    self.style.configure(
        "blue.Horizontal.TScale",
        troughcolor="#d1e0ff",  # Light blue
        background="#4287f5",   # Medium blue
        bordercolor="#1c56c5"   # Dark blue
    )
    
    # Create blue button style
    self.style.configure(
        "blue.TButton",
        background="#4287f5",
        relief="raised"
    )
    
    # Create red depolarization slider style
    self.style.configure(
        "red.Horizontal.TScale",
        troughcolor="#ffd1d1",  # Light red
        background="#f54242",   # Medium red
        bordercolor="#c51c1c"   # Dark red
    )
    
    # Create red button style
    self.style.configure(
        "red.TButton",
        background="#f54242",
        relief="raised"
    )

def start_drag(self, slider, slider_type):
    """Start dragging a slider handle."""
    # Store the slider being dragged
    self.current_drag = {
        'slider': slider,
        'type': slider_type
    }

def drag_slider(self, event, slider, slider_type):
    """Handle dragging of slider handle."""
    if not hasattr(self, 'current_drag'):
        return
        
    # Calculate new slider value based on mouse position
    x = event.x_root
    slider_width = slider.winfo_width()
    slider_x = slider.winfo_rootx()
    
    # Calculate relative position (0 to 1)
    relative_pos = max(0, min(1, (x - slider_x) / slider_width))
    
    # Convert to slider value
    min_val = float(slider.cget('from'))
    max_val = float(slider.cget('to'))
    new_val = min_val + relative_pos * (max_val - min_val)
    
    # Apply snapping to grid of 5
    snapped_val = round(new_val / 5) * 5
    
    # Update slider value
    slider_var = getattr(self, slider_type)
    slider_var.set(snapped_val)
    
    # Get the corresponding display label
    label_name = f"{slider_type}_display"
    if hasattr(self, label_name):
        label = getattr(self, label_name)
        # Update display
        self.update_slider_display(snapped_val, label, slider_type)

def update_handle_positions(self):
    """Update the positions of all handle buttons to match slider values."""
    for slider_type in ['hyperpol_start', 'hyperpol_end', 'depol_start', 'depol_end']:
        handle_name = f"{slider_type}_handle"
        slider_name = f"{slider_type}_slider"
        
        if hasattr(self, handle_name) and hasattr(self, slider_name):
            handle = getattr(self, handle_name)
            slider = getattr(self, slider_name)
            
            # Calculate handle position based on slider value
            slider_var = getattr(self, slider_type)
            value = slider_var.get()
            min_val = float(slider.cget('from'))
            max_val = float(slider.cget('to'))
            
            # Calculate relative position
            if max_val > min_val:
                relative_pos = (value - min_val) / (max_val - min_val)
                # Update handle position
                handle.place(relx=relative_pos, rely=0.5, anchor='center')