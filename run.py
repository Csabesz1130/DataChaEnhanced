import os
import sys

# Optimize path setup
current_dir = os.path.dirname(os.path.abspath(__file__))

# For PyInstaller, we need to handle the path differently
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = sys._MEIPASS
    src_path = os.path.join(application_path, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if application_path not in sys.path:
        sys.path.insert(0, application_path)
else:
    # Running as script - only add paths if not already present
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

# Run the application
if __name__ == "__main__":
    try:
        from src.main import main
        main()
    except ImportError as e:
        # Simplified error handling
        print(f"Import error: {e}")
        print(f"Python path: {sys.path[:3]}...")  # Show only first 3 paths
        print(f"Current directory: {current_dir}")
        
        # Try alternative import without verbose output
        try:
            import main
            main.main()
        except ImportError:
            print("Failed to import main module. Please check your installation.")
            if not getattr(sys, 'frozen', False):
                input("Press Enter to exit...")
            sys.exit(1)