"""
Application Integration Module for Curve Fitting
================================================
Location: src/gui/app_integration.py

This module integrates curve fitting into DataChaEnhanced and fixes window sizing.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger(__name__)

def integrate_curve_fitting(main_app):
    """
    Integrate curve fitting functionality into the main application.
    Call this after the main app is initialized.
    
    Args:
        main_app: The MainApp instance from DataChaEnhanced
    """
    logger.info("Integrating curve fitting functionality...")
    
    # Import curve fitting components
    from src.gui.curve_fitting_gui import CurveFittingPanel
    
    # Find the ActionPotentialTab
    if hasattr(main_app, 'action_potential_tab'):
        ap_tab = main_app.action_potential_tab
        
        # Add curve fitting panel to the ActionPotentialTab
        if hasattr(ap_tab, 'main_frame'):
            # Create the curve fitting panel
            ap_tab.curve_fitting_panel = CurveFittingPanel(ap_tab.main_frame, main_app)
            
            # Store reference in main app
            main_app.curve_fitting_panel = ap_tab.curve_fitting_panel
            
            # Hook into the plot update system
            original_update = main_app.update_plot_with_processed_data
            
            def enhanced_update(*args, **kwargs):
                # Call original update
                result = original_update(*args, **kwargs)
                
                # Initialize/update curve fitting manager
                if hasattr(main_app, 'curve_fitting_panel'):
                    panel = main_app.curve_fitting_panel
                    
                    # Get the plot from ActionPotentialTab
                    if hasattr(ap_tab, 'fig') and hasattr(ap_tab, 'ax'):
                        if not panel.fitting_manager:
                            panel.initialize_fitting_manager(ap_tab.fig, ap_tab.ax)
                        else:
                            panel.update_curve_data()
                
                return result
            
            # Replace the update method
            main_app.update_plot_with_processed_data = enhanced_update
            
            logger.info("✅ Curve fitting integrated into ActionPotentialTab")
        else:
            logger.warning("Could not find main_frame in ActionPotentialTab")
    else:
        logger.warning("ActionPotentialTab not found in main application")

def fix_window_sizing(main_app):
    """
    Fix window sizing issues to allow proper zooming and resizing.
    
    Args:
        main_app: The MainApp instance
    """
    logger.info("Fixing window sizing issues...")
    
    try:
        # Get the root window
        root = main_app.root if hasattr(main_app, 'root') else main_app
        
        # Remove any size constraints
        root.minsize(800, 600)  # Set reasonable minimum size
        root.maxsize(root.winfo_screenwidth(), root.winfo_screenheight())
        
        # Enable resizing
        root.resizable(True, True)
        
        # Set initial size to 80% of screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Bind zoom shortcuts
        def zoom_in(event=None):
            current_size = root.winfo_width(), root.winfo_height()
            new_width = min(int(current_size[0] * 1.1), screen_width)
            new_height = min(int(current_size[1] * 1.1), screen_height)
            root.geometry(f"{new_width}x{new_height}")
        
        def zoom_out(event=None):
            current_size = root.winfo_width(), root.winfo_height()
            new_width = max(int(current_size[0] * 0.9), 800)
            new_height = max(int(current_size[1] * 0.9), 600)
            root.geometry(f"{new_width}x{new_height}")
        
        def reset_zoom(event=None):
            root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Bind keyboard shortcuts
        root.bind('<Control-plus>', zoom_in)
        root.bind('<Control-equal>', zoom_in)  # For keyboards without numpad
        root.bind('<Control-minus>', zoom_out)
        root.bind('<Control-0>', reset_zoom)
        
        # Add zoom controls to menu if exists
        if hasattr(main_app, 'menubar'):
            view_menu = tk.Menu(main_app.menubar, tearoff=0)
            view_menu.add_command(label="Zoom In (Ctrl++)", command=zoom_in)
            view_menu.add_command(label="Zoom Out (Ctrl+-)", command=zoom_out)
            view_menu.add_command(label="Reset Zoom (Ctrl+0)", command=reset_zoom)
            view_menu.add_separator()
            view_menu.add_command(label="Fullscreen (F11)", 
                                command=lambda: root.attributes('-fullscreen', 
                                                               not root.attributes('-fullscreen')))
            main_app.menubar.add_cascade(label="View", menu=view_menu)
        
        # Bind F11 for fullscreen toggle
        root.bind('<F11>', lambda e: root.attributes('-fullscreen', 
                                                     not root.attributes('-fullscreen')))
        root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
        
        # Fix matplotlib figure sizing in ActionPotentialTab
        if hasattr(main_app, 'action_potential_tab'):
            ap_tab = main_app.action_potential_tab
            if hasattr(ap_tab, 'canvas'):
                # Make canvas resizable
                ap_tab.canvas.get_tk_widget().pack(fill='both', expand=True)
                
                # Update figure size on window resize
                def on_resize(event=None):
                    if hasattr(ap_tab, 'fig'):
                        # Get canvas size in inches
                        width = ap_tab.canvas.get_tk_widget().winfo_width() / 100
                        height = ap_tab.canvas.get_tk_widget().winfo_height() / 100
                        if width > 0 and height > 0:
                            ap_tab.fig.set_size_inches(width, height, forward=True)
                
                ap_tab.canvas.get_tk_widget().bind('<Configure>', on_resize)
        
        logger.info("✅ Window sizing fixed")
        
    except Exception as e:
        logger.error(f"Error fixing window sizing: {str(e)}")

def enhance_application(main_app):
    """
    Main enhancement function that adds all new features.
    
    Args:
        main_app: The MainApp instance from DataChaEnhanced
    """
    try:
        # Fix window sizing issues
        fix_window_sizing(main_app)
        
        # Integrate curve fitting
        integrate_curve_fitting(main_app)
        
        logger.info("✅ Application enhancements completed")
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing application: {str(e)}")
        return False

# Monkey patch for existing MainApp
def patch_main_app():
    """
    Patch the existing MainApp class to include enhancements.
    Add this to the main app initialization.
    """
    try:
        from src.gui.app import MainApp
        
        # Store original __init__
        original_init = MainApp.__init__
        
        def enhanced_init(self, *args, **kwargs):
            # Call original initialization
            original_init(self, *args, **kwargs)
            
            # Add enhancements after initialization
            self.root.after(100, lambda: enhance_application(self))
        
        # Replace __init__
        MainApp.__init__ = enhanced_init
        
        logger.info("✅ MainApp patched with enhancements")
        
    except ImportError:
        logger.error("Could not import MainApp for patching")