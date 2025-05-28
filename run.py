import os
import sys

# Add both current directory and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

# For PyInstaller, we need to handle the path differently
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = sys._MEIPASS
    src_path = os.path.join(application_path, 'src')
    sys.path.insert(0, src_path)
    sys.path.insert(0, application_path)
else:
    # Running as script
    sys.path.insert(0, current_dir)
    sys.path.insert(0, src_dir)

# Now import and run the main function
try:
    from src.main import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {current_dir}")
    print(f"Src directory: {src_dir}")
    
    # Try alternative import
    try:
        import main
        if __name__ == "__main__":
            main.main()
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        input("Press Enter to exit...")
        sys.exit(1)