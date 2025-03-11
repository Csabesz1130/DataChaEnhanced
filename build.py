import os
import sys
import shutil
from pathlib import Path
from PyInstaller.__main__ import run

def clean_dist():
    """Clean previous build artifacts"""
    print("Cleaning previous builds...")
    paths_to_clean = ['build', 'dist']
    for path in paths_to_clean:
        if os.path.exists(path):
            shutil.rmtree(path)
    spec_file = 'SignalAnalyzer.spec'
    if os.path.exists(spec_file):
        os.remove(spec_file)

# Add to build.py
def zip_application():
    """Create a ZIP archive of the application"""
    print("Creating ZIP archive...")
    shutil.make_archive('dist/SignalAnalyzer', 'zip', 'dist/SignalAnalyzer')

# Then call this in main() after copy_additional_files()

def create_executable():
    """Create the executable using PyInstaller"""
    print("Creating executable...")

    # Point to src/run.py
    script_path = os.path.join(os.path.dirname(__file__), "src", "run.py")

    # Define PyInstaller arguments (use the absolute path to run.py)
    args = [
        script_path,
        "--name=SignalAnalyzer",
        "--onedir",
        "--windowed",
        "--icon=assets/icon.ico",       # Adjust or remove if you don't have an icon
        "--add-data=assets;assets",     # Include the assets folder in the build
        f"--paths={os.path.join(os.path.dirname(__file__), 'src')}",
        "--hidden-import=tkinter",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=pandas",
        "--hidden-import=PyWavelets"
    ]

    # Run PyInstaller
    run(args)

def copy_additional_files():
    """Copy additional required files to the dist directory"""
    print("Copying additional files...")
    dist_dir = Path('dist/SignalAnalyzer')
    assets_dir = dist_dir / 'assets'
    assets_dir.mkdir(exist_ok=True)
    # Example of copying a config file:
    # shutil.copy2('config.ini', dist_dir)

def main():
    try:
        clean_dist()
        create_executable()
        copy_additional_files()
        # Call zip_application after copy_additional_files
        zip_application()
        print("Build completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during build: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
