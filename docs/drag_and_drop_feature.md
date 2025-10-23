# Drag and Drop Feature Documentation

## Overview

The drag and drop feature in DataChaEnhanced allows users to load ATF (Axon Text Format) files by simply dragging them from the file system onto the application's plot area. This feature is built using the `tkinterdnd2` library and includes comprehensive error handling, fallback mechanisms, and integration with the hot reload system.

## System Architecture

### Core Components

1. **Main Entry Point** (`src/main.py`)
   - Initializes the application with drag and drop support
   - Creates TkinterDnD.Tk() root window if tkinterdnd2 is available
   - Falls back to standard tk.Tk() if library is missing

2. **Application Class** (`src/gui/app.py`)
   - Contains the main drag and drop implementation
   - Manages drag and drop state and hot reload coordination
   - Handles file validation and loading

3. **Hot Reload Integration** (`src/utils/hot_reload.py`)
   - Coordinates with drag and drop to prevent conflicts
   - Manages file system monitoring during drag operations

### Dependencies

#### Required Libraries:
```python
# Core drag and drop functionality
from tkinterdnd2 import DND_FILES, TkinterDnD

# Standard libraries
import tkinter as tk
import os
import gc
```

#### Optional Libraries:
```python
# For hot reload functionality (development)
from src.utils.hot_reload import initialize_hot_reload, stop_hot_reload
```

## Implementation Details

### 1. Initialization and Setup

#### Main Application Entry (`src/main.py`):
```python
def _check_drag_drop_availability():
    """Check if tkinterdnd2 is available"""
    global DRAG_DROP_AVAILABLE, TkinterDnD
    try:
        from tkinterdnd2 import TkinterDnD
        DRAG_DROP_AVAILABLE = True
        return True
    except ImportError:
        DRAG_DROP_AVAILABLE = False
        return False

def main():
    """Main application entry point"""
    # Create the main window with drag and drop support if available
    if _check_drag_drop_availability():
        root = TkinterDnD.Tk()  # Enhanced Tk with drag and drop
    else:
        root = tk.Tk()  # Standard Tk fallback
```

#### Application Initialization (`src/gui/app.py`):
```python
class SignalAnalyzerApp:
    def __init__(self, master):
        # Drag and drop state management
        self._drag_drop_active = False  # Flag to prevent reload during drag operations
        self._drag_drop_setup_done = False  # Prevent multiple setups
        
        # Setup drag and drop after initialization
        self.master.after(500, self.setup_drag_and_drop)
```

### 2. Drag and Drop Setup

#### Core Setup Function:
```python
def setup_drag_and_drop(self):
    """Setup drag and drop functionality for file loading."""
    try:
        # Prevent multiple setups
        if hasattr(self, '_drag_drop_setup_done') and self._drag_drop_setup_done:
            app_logger.info("🖱️ Drag and drop already set up, skipping...")
            return
            
        # Recheck availability in case of hot reload
        if not _check_drag_drop_availability():
            app_logger.warning("Drag and drop not available - using double-click fallback")
            return

        # Check if plot is initialized
        if not hasattr(self, 'canvas') or not self.canvas:
            app_logger.info("🖱️ Plot not initialized yet, retrying in 1 second...")
            self.master.after(1000, self.setup_drag_and_drop)
            return

        # Get the canvas widget and plot frame
        canvas_widget = self.canvas.get_tk_widget()
        plot_frame = self.plot_frame
        
        # Enable drag and drop on both canvas and plot frame for better coverage
        for widget in [canvas_widget, plot_frame]:
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_file_drop)
                widget.dnd_bind("<<DragEnter>>", self._on_drag_enter_dnd)
                widget.dnd_bind("<<DragLeave>>", self._on_drag_leave_dnd)
            except Exception as e:
                app_logger.error(f"Could not register drop target on {widget}: {e}")

        # Add visual feedback and double-click to load
        canvas_widget.bind("<Enter>", self._on_drag_enter)
        canvas_widget.bind("<Leave>", self._on_drag_leave)
        canvas_widget.bind("<Double-Button-1>", self._on_canvas_double_click)

        # Mark as done to prevent multiple setups
        self._drag_drop_setup_done = True
        
    except Exception as e:
        app_logger.error(f"Error setting up drag and drop: {e}")
        # Try again after a short delay
        self.master.after(1000, self.setup_drag_and_drop)
```

### 3. Event Handling

#### File Drop Handler:
```python
def _on_file_drop(self, event):
    """Handle file drop event (tkinterdnd2)."""
    try:
        app_logger.info(f"🖱️ Drop event triggered with data: {event.data}")
        
        # Parse dropped files - handle different formats
        files = []
        if hasattr(event, 'data') and event.data:
            # Split by whitespace and clean up
            raw_files = event.data.split()
            for f in raw_files:
                # Remove braces and quotes if present
                clean_file = f.strip("{}").strip('"').strip("'")
                if clean_file:
                    files.append(clean_file)
        
        if files:
            # Filter for ATF files only
            atf_files = [f for f in files if f.lower().endswith('.atf')]
            if atf_files:
                # Take the first ATF file
                filepath = atf_files[0]
                app_logger.info(f"🖱️ ATF file dropped: {filepath}")
                
                # Re-enable hot reload after file processing
                self._drag_drop_active = False
                
                # Load the file
                self._load_dropped_file(filepath)
            else:
                app_logger.warning("🖱️ No ATF files found in drop - ignoring")
                self._drag_drop_active = False
                messagebox.showwarning(
                    "Invalid File Type",
                    "Please drop an ATF file (.atf extension).\n\n"
                    f"Files dropped: {', '.join(files)}"
                )
        else:
            app_logger.warning("🖱️ No files found in drop event")
            self._drag_drop_active = False
            
    except Exception as e:
        app_logger.error(f"Error handling file drop: {e}")
        self._drag_drop_active = False
```

#### Drag Enter/Leave Handlers:
```python
def _on_drag_enter_dnd(self, event):
    """Handle tkinterdnd2 drag enter event."""
    try:
        # Disable hot reload during drag operations
        self._drag_drop_active = True
        app_logger.info("🖱️ Drag enter - hot reload disabled")
    except Exception as e:
        app_logger.error(f"Error in drag enter: {e}")

def _on_drag_leave_dnd(self, event):
    """Handle tkinterdnd2 drag leave event."""
    try:
        # Re-enable hot reload after drag operations
        self._drag_drop_active = False
        app_logger.info("🖱️ Drag leave - hot reload re-enabled")
    except Exception as e:
        app_logger.error(f"Error in drag leave: {e}")
```

### 4. File Loading Process

#### Dropped File Handler:
```python
def _load_dropped_file(self, filepath):
    """Load a dropped file."""
    try:
        # Validate file extension
        if not filepath.lower().endswith(".atf"):
            app_logger.warning(f"⚠️ Dropped file is not an ATF file: {filepath}")
            messagebox.showwarning(
                "Invalid File Type",
                "Please drop an ATF file (.atf extension).\n\n"
                f"Dropped file: {filepath}",
            )
            return

        # Check if file exists
        if not os.path.exists(filepath):
            app_logger.error(f"❌ Dropped file does not exist: {filepath}")
            messagebox.showerror(
                "File Not Found", f"The file does not exist:\n{filepath}"
            )
            return

        app_logger.info(f"📁 Loading dropped file: {filepath}")

        # Ensure plot is initialized before loading data
        self._ensure_plot_initialized()

        # Clear existing data
        self.clear_all_data()
        self.clear_saved_limits()

        # Force cleanup before loading new data
        gc.collect()

        # Store current file path
        self.current_file = filepath

        # Load with memory optimization
        atf_handler = ATFHandler(filepath)
        atf_handler.load_atf()

        # Get data and store using properties
        new_time_data = atf_handler.get_column("Time")
        new_data = atf_handler.get_column("#1")

        self.time_data = new_time_data
        self.data = new_data
        self.filtered_data = new_data.copy()

        # Clean up handler
        del atf_handler
        gc.collect()

        # Update view limits and analysis tabs
        self.view_tab.update_limits(
            t_min=self.time_data[0],
            t_max=self.time_data[-1],
            v_min=np.min(self.data),
            v_max=np.max(self.data),
        )

        self.analysis_tab.update_data(self.data, self.time_data)

        # Clear Action Potential tab to remove previous analysis results
        if hasattr(self, "action_potential_tab") and self.action_potential_tab:
            self.action_potential_tab.reset()
            app_logger.info("Action Potential tab cleared for new file")

        # Extract voltage from filename and initialize processor if found
        voltage = ActionPotentialProcessor.parse_voltage_from_filename(filepath)
        if voltage is not None:
            app_logger.info(f"Detected V2 voltage from filename: {voltage} mV")
            self.action_potential_tab.V2.set(voltage)

        self.update_plot(force_full_range=True)

        # Show success message
        filename = os.path.basename(filepath)
        messagebox.showinfo("File Loaded", f"Successfully loaded:\n{filename}")

        app_logger.info(f"File loaded successfully via drag and drop: {filename}")

    except Exception as e:
        app_logger.error(f"Error loading dropped file: {e}")
        messagebox.showerror("Load Error", f"Failed to load file:\n{str(e)}")
```

## Common Issues and Solutions

### 1. Library Import Issues

#### Problem: `tkinterdnd2` not available
```python
# Error: ImportError: No module named 'tkinterdnd2'
```

**Solution:**
```bash
# Install tkinterdnd2
pip install tkinterdnd2

# Or for specific Python version
pip3 install tkinterdnd2
```

**Fallback Implementation:**
```python
def _check_drag_drop_availability():
    """Check if tkinterdnd2 is available and set global variables"""
    global DRAG_DROP_AVAILABLE, DND_FILES, TkinterDnD
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        DRAG_DROP_AVAILABLE = True
        app_logger.info("tkinterdnd2 imported successfully - drag and drop enabled")
        return True
    except ImportError as e:
        DRAG_DROP_AVAILABLE = False
        app_logger.warning(f"tkinterdnd2 not available - drag and drop disabled: {e}")
        return False
```

### 2. Multiple Setup Issues

#### Problem: Drag and drop setup called multiple times
```python
# Error: Multiple drop target registrations
```

**Solution:**
```python
def setup_drag_and_drop(self):
    """Setup drag and drop functionality for file loading."""
    try:
        # Prevent multiple setups
        if hasattr(self, '_drag_drop_setup_done') and self._drag_drop_setup_done:
            app_logger.info("🖱️ Drag and drop already set up, skipping...")
            return
        
        # ... setup code ...
        
        # Mark as done to prevent multiple setups
        self._drag_drop_setup_done = True
```

### 3. Hot Reload Conflicts

#### Problem: Hot reload triggers during drag operations
```python
# Error: Hot reload interferes with drag and drop
```

**Solution:**
```python
def _on_drag_enter_dnd(self, event):
    """Handle tkinterdnd2 drag enter event."""
    try:
        # Disable hot reload during drag operations
        self._drag_drop_active = True
        app_logger.info("🖱️ Drag enter - hot reload disabled")
    except Exception as e:
        app_logger.error(f"Error in drag enter: {e}")

def _on_drag_leave_dnd(self, event):
    """Handle tkinterdnd2 drag leave event."""
    try:
        # Re-enable hot reload after drag operations
        self._drag_drop_active = False
        app_logger.info("🖱️ Drag leave - hot reload re-enabled")
    except Exception as e:
        app_logger.error(f"Error in drag leave: {e}")
```

### 4. File Path Parsing Issues

#### Problem: Incorrect file path parsing from drop event
```python
# Error: File path contains extra characters or quotes
```

**Solution:**
```python
def _on_file_drop(self, event):
    """Handle file drop event (tkinterdnd2)."""
    try:
        # Parse dropped files - handle different formats
        files = []
        if hasattr(event, 'data') and event.data:
            # Split by whitespace and clean up
            raw_files = event.data.split()
            for f in raw_files:
                # Remove braces and quotes if present
                clean_file = f.strip("{}").strip('"').strip("'")
                if clean_file:
                    files.append(clean_file)
```

### 5. Canvas Initialization Issues

#### Problem: Drag and drop setup before canvas is ready
```python
# Error: Canvas not initialized when setting up drag and drop
```

**Solution:**
```python
def setup_drag_and_drop(self):
    """Setup drag and drop functionality for file loading."""
    try:
        # Check if plot is initialized
        if not hasattr(self, 'canvas') or not self.canvas:
            app_logger.info("🖱️ Plot not initialized yet, retrying in 1 second...")
            self.master.after(1000, self.setup_drag_and_drop)
            return
```

## What to Avoid

### 1. Multiple Drop Target Registration

**❌ DON'T:**
```python
# Don't register multiple times without checking
widget.drop_target_register(DND_FILES)
widget.drop_target_register(DND_FILES)  # This will cause errors
```

**✅ DO:**
```python
# Check if already registered
if not hasattr(widget, '_drop_target_registered'):
    widget.drop_target_register(DND_FILES)
    widget._drop_target_registered = True
```

### 2. Ignoring Hot Reload State

**❌ DON'T:**
```python
# Don't ignore hot reload state during drag operations
def _on_file_drop(self, event):
    # Process file without checking hot reload state
    self.load_file(event.data)
```

**✅ DO:**
```python
# Always check and manage hot reload state
def _on_file_drop(self, event):
    try:
        # Process file
        self.load_file(event.data)
    finally:
        # Always re-enable hot reload
        self._drag_drop_active = False
```

### 3. Missing Error Handling

**❌ DON'T:**
```python
# Don't ignore errors in drag and drop setup
def setup_drag_and_drop(self):
    widget.drop_target_register(DND_FILES)  # No error handling
```

**✅ DO:**
```python
# Always handle errors gracefully
def setup_drag_and_drop(self):
    try:
        widget.drop_target_register(DND_FILES)
    except Exception as e:
        app_logger.error(f"Could not register drop target: {e}")
        # Implement fallback or retry mechanism
```

### 4. Hardcoded File Extensions

**❌ DON'T:**
```python
# Don't hardcode file extensions
if filepath.endswith('.atf'):
    self.load_file(filepath)
```

**✅ DO:**
```python
# Use case-insensitive comparison
if filepath.lower().endswith('.atf'):
    self.load_file(filepath)
```

### 5. Memory Leaks

**❌ DON'T:**
```python
# Don't forget to clean up resources
def _load_dropped_file(self, filepath):
    atf_handler = ATFHandler(filepath)
    atf_handler.load_atf()
    # Missing cleanup
```

**✅ DO:**
```python
# Always clean up resources
def _load_dropped_file(self, filepath):
    atf_handler = ATFHandler(filepath)
    atf_handler.load_atf()
    # ... process data ...
    
    # Clean up handler
    del atf_handler
    gc.collect()
```

## Best Practices

### 1. Graceful Degradation

```python
def setup_drag_and_drop(self):
    """Setup drag and drop with graceful degradation."""
    try:
        if not _check_drag_drop_availability():
            app_logger.warning("Drag and drop not available - using double-click fallback")
            # Setup double-click fallback
            self._setup_double_click_fallback()
            return
        # ... setup drag and drop ...
    except Exception as e:
        app_logger.error(f"Error setting up drag and drop: {e}")
        # Fallback to double-click
        self._setup_double_click_fallback()
```

### 2. State Management

```python
class SignalAnalyzerApp:
    def __init__(self, master):
        # Clear state management
        self._drag_drop_active = False
        self._drag_drop_setup_done = False
        
    def _on_drag_enter_dnd(self, event):
        """Handle drag enter with proper state management."""
        self._drag_drop_active = True
        
    def _on_drag_leave_dnd(self, event):
        """Handle drag leave with proper state management."""
        self._drag_drop_active = False
        
    def _on_file_drop(self, event):
        """Handle file drop with proper state management."""
        try:
            # Process file
            self._process_dropped_file(event.data)
        finally:
            # Always reset state
            self._drag_drop_active = False
```

### 3. Comprehensive Error Handling

```python
def _on_file_drop(self, event):
    """Handle file drop with comprehensive error handling."""
    try:
        app_logger.info(f"🖱️ Drop event triggered with data: {event.data}")
        
        # Parse and validate files
        files = self._parse_dropped_files(event.data)
        if not files:
            app_logger.warning("🖱️ No files found in drop event")
            return
            
        # Filter for supported files
        supported_files = self._filter_supported_files(files)
        if not supported_files:
            self._show_invalid_file_warning(files)
            return
            
        # Load the first supported file
        self._load_dropped_file(supported_files[0])
        
    except Exception as e:
        app_logger.error(f"Error handling file drop: {e}")
        messagebox.showerror("Drop Error", f"Failed to process dropped files:\n{str(e)}")
    finally:
        # Always reset state
        self._drag_drop_active = False
```

### 4. User Feedback

```python
def _add_drag_drop_tooltip(self, widget):
    """Add visual feedback for drag and drop functionality."""
    try:
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(
                tooltip,
                text="Double-click to load file",
                bg="lightyellow",
                relief="solid",
                borderwidth=1,
            )
            label.pack()
            widget.tooltip = tooltip

        def hide_tooltip(event):
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                delattr(widget, "tooltip")

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
    except Exception as e:
        app_logger.debug(f"Tooltip setup failed: {e}")
```

## Testing and Debugging

### 1. Debug Logging

```python
def setup_drag_and_drop(self):
    """Setup drag and drop with comprehensive logging."""
    try:
        app_logger.info("🖱️ Setting up drag and drop...")
        
        # Log availability check
        if not _check_drag_drop_availability():
            app_logger.warning("Drag and drop not available - using double-click fallback")
            return

        # Log canvas state
        if not hasattr(self, 'canvas') or not self.canvas:
            app_logger.info("🖱️ Plot not initialized yet, retrying in 1 second...")
            return
            
        # Log widget registration
        canvas_widget = self.canvas.get_tk_widget()
        plot_frame = self.plot_frame
        
        app_logger.info(f"🖱️ Canvas widget: {canvas_widget}")
        app_logger.info(f"🖱️ Plot frame: {plot_frame}")
        
        # Register drop targets
        for widget in [canvas_widget, plot_frame]:
            try:
                app_logger.info(f"🖱️ Registering drop target on {widget}")
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_file_drop)
                app_logger.info(f"🖱️ Drop target registered successfully on {widget}")
            except Exception as e:
                app_logger.error(f"Could not register drop target on {widget}: {e}")
        
        app_logger.info("🖱️ Drag and drop setup complete for ATF files")
        
    except Exception as e:
        app_logger.error(f"Error setting up drag and drop: {e}")
```

### 2. Testing Drag and Drop

```python
def _test_drag_drop_setup(self):
    """Test method to manually trigger drag and drop setup."""
    try:
        app_logger.info("🧪 Manual drag and drop test triggered")
        self.setup_drag_and_drop()
    except Exception as e:
        app_logger.error(f"Manual drag and drop test failed: {e}")
```

## Performance Considerations

### 1. Memory Management

```python
def _load_dropped_file(self, filepath):
    """Load dropped file with memory optimization."""
    try:
        # Force cleanup before loading new data
        gc.collect()
        
        # Load with memory optimization
        atf_handler = ATFHandler(filepath)
        atf_handler.load_atf()
        
        # Process data
        new_time_data = atf_handler.get_column("Time")
        new_data = atf_handler.get_column("#1")
        
        # Store data
        self.time_data = new_time_data
        self.data = new_data
        self.filtered_data = new_data.copy()
        
        # Clean up handler
        del atf_handler
        gc.collect()
        
    except Exception as e:
        app_logger.error(f"Error loading dropped file: {e}")
```

### 2. State Management Efficiency

```python
def _on_drag_enter_dnd(self, event):
    """Handle drag enter efficiently."""
    # Only set flag if not already set
    if not self._drag_drop_active:
        self._drag_drop_active = True
        app_logger.info("🖱️ Drag enter - hot reload disabled")

def _on_drag_leave_dnd(self, event):
    """Handle drag leave efficiently."""
    # Only reset flag if currently set
    if self._drag_drop_active:
        self._drag_drop_active = False
        app_logger.info("🖱️ Drag leave - hot reload re-enabled")
```

## Conclusion

The drag and drop feature in DataChaEnhanced provides a user-friendly way to load ATF files while maintaining robust error handling and integration with the hot reload system. Key aspects include:

- **Graceful Degradation**: Falls back to double-click if tkinterdnd2 is unavailable
- **State Management**: Proper coordination with hot reload system
- **Error Handling**: Comprehensive error handling and user feedback
- **Memory Management**: Efficient resource cleanup
- **User Experience**: Visual feedback and clear error messages

The implementation follows best practices for tkinterdnd2 integration and provides a solid foundation for file loading functionality.
