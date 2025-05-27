import os
import sys
import shutil
import subprocess
from pathlib import Path
from PyInstaller.__main__ import run
import zipfile
import json
from datetime import datetime

class SignalAnalyzerBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.assets_dir = self.project_root / "assets"
        
    def clean_previous_builds(self):
        """Clean previous build artifacts"""
        print(">> Cleaning previous builds...")
        paths_to_clean = [self.build_dir, self.dist_dir]
        
        for path in paths_to_clean:
            if path.exists():
                shutil.rmtree(path)
                print(f"   Removed: {path}")
        
        # Remove spec file
        spec_file = self.project_root / 'SignalAnalyzer.spec'
        if spec_file.exists():
            spec_file.unlink()
            print(f"   Removed: {spec_file}")

    def ensure_assets_exist(self):
        """Create assets directory and default icon if they don't exist"""
        print(">> Ensuring assets exist...")
        
        if not self.assets_dir.exists():
            self.assets_dir.mkdir()
            print(f"   Created: {self.assets_dir}")
        
        # Create a simple default icon if none exists
        icon_path = self.assets_dir / "icon.ico"
        if not icon_path.exists():
            print("   Creating default icon...")
            try:
                from PIL import Image, ImageDraw
                # Create a simple 32x32 icon
                img = Image.new('RGBA', (32, 32), (70, 130, 180, 255))
                draw = ImageDraw.Draw(img)
                draw.ellipse([8, 8, 24, 24], fill=(255, 255, 255, 255))
                img.save(icon_path, format='ICO')
                print(f"   Created default icon: {icon_path}")
            except ImportError:
                print("   PIL not available, skipping icon creation")
                return None
        
        return icon_path

    def get_all_hidden_imports(self):
        """Get comprehensive list of hidden imports"""
        hidden_imports = [
            # Core GUI
            'tkinter',
            'tkinter.ttk',
            'tkinter.filedialog',
            'tkinter.messagebox',
            
            # Scientific computing
            'numpy',
            'scipy',
            'scipy.signal',
            'scipy.fft',
            'matplotlib',
            'matplotlib.backends.backend_tkagg',
            'matplotlib.figure',
            'matplotlib.pyplot',
            'pandas',
            
            # Wavelets
            'PyWavelets',
            'pywt',
            
            # Machine learning (if used)
            'sklearn',
            'sklearn.linear_model',
            
            # Imaging (for icon creation)
            'PIL',
            'PIL.Image',
            'PIL.ImageDraw',
            
            # Your application modules - CRITICAL ADDITION
            'src',
            'src.main',
            'src.gui',
            'src.gui.app',
            'src.gui.filter_tab',
            'src.gui.analysis_tab',
            'src.gui.view_tab',
            'src.gui.action_potential_tab',
            'src.gui.window_manager',
            'src.io_utils',
            'src.io_utils.io_utils',
            'src.filtering',
            'src.filtering.filtering',
            'src.utils',
            'src.utils.logger',
            'src.analysis',
            'src.analysis.action_potential',
        ]
        
        return hidden_imports

    def create_version_info(self):
        """Create version info for the executable"""
        version_info = {
            "version": "1.0.2",
            "build_date": datetime.now().isoformat(),
            "description": "Signal Analyzer - Advanced Signal Processing Tool",
            "company": "Signal Analysis Lab",
            "product": "Signal Analyzer"
        }
        
        version_file = self.project_root / "version_info.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2)
        
        return version_info

    def create_executable(self):
        """Create the executable using PyInstaller with comprehensive settings"""
        print(">> Creating executable...")
        
        # Ensure run.py exists
        run_script = self.project_root / "run.py"
        if not run_script.exists():
            print("ERROR: run.py not found!")
            return False
        
        # Get icon path
        icon_path = self.ensure_assets_exist()
        
        # Create version info
        version_info = self.create_version_info()
        
        # Build PyInstaller arguments
        args = [
            str(run_script),
            "--name=SignalAnalyzer",
            "--onedir",  # Create a folder distribution
            "--windowed",  # No console window
            "--clean",  # Clean cache
            f"--distpath={self.dist_dir}",
            f"--workpath={self.build_dir}",
            "--noconfirm",  # Don't ask for confirmation
        ]
        
        # Add icon if available
        if icon_path and icon_path.exists():
            args.append(f"--icon={icon_path}")
        
        # CRITICAL: Add entire src directory as data
        if self.src_dir.exists():
            args.append(f"--add-data={self.src_dir};src")
            print(f"   Added source directory: {self.src_dir}")
        
        # Add assets directory
        if self.assets_dir.exists():
            args.append(f"--add-data={self.assets_dir};assets")
        
        # Add source path - CRITICAL for imports
        args.append(f"--paths={self.src_dir}")
        args.append(f"--paths={self.project_root}")
        
        # Add hidden imports - INCLUDING ALL YOUR MODULES
        for import_name in self.get_all_hidden_imports():
            args.append(f"--hidden-import={import_name}")
        
        # Additional options for better compatibility
        args.extend([
            "--noupx",  # Don't use UPX compression
            "--exclude-module=PIL.ImageQt",
            "--exclude-module=PyQt5",
            "--exclude-module=PyQt6",
            "--collect-all=src",  # CRITICAL: Collect all submodules from src
        ])
        
        print(f"   Running PyInstaller...")
        print(f"   Main script: {run_script}")
        print(f"   Source directory: {self.src_dir}")
        
        try:
            run(args)
            print("SUCCESS: Executable created successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Creating executable failed: {e}")
            return False

    def copy_additional_files(self):
        """Copy additional required files to the dist directory"""
        print(">> Copying additional files...")
        
        exe_dir = self.dist_dir / 'SignalAnalyzer'
        if not exe_dir.exists():
            print("ERROR: Executable directory not found!")
            return False
        
        # Copy version info
        version_file = self.project_root / "version_info.json"
        if version_file.exists():
            shutil.copy2(version_file, exe_dir)
            print(f"   Copied: {version_file.name}")
        
        # Copy README if exists
        readme_files = ['README.md', 'README.txt', 'readme.txt']
        for readme in readme_files:
            readme_path = self.project_root / readme
            if readme_path.exists():
                shutil.copy2(readme_path, exe_dir)
                print(f"   Copied: {readme}")
                break
        
        # Create a simple user guide
        user_guide = exe_dir / "User_Guide.txt"
        with open(user_guide, 'w', encoding='utf-8') as f:
            f.write("""Signal Analyzer - User Guide
=============================

Getting Started:
1. Double-click SignalAnalyzer.exe to start the application
2. Use File > Load to load your ATF data files
3. Apply filters using the Filter tab
4. Analyze signals using the Analysis tab
5. Export results when done

For support, contact your instructor.

Build Date: {build_date}
Version: {version}
""".format(
    build_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    version="1.0.2"
))
        print(f"   Created: {user_guide.name}")
        
        return True

    def create_installer_package(self):
        """Create a ZIP package for easy distribution"""
        print(">> Creating distribution package...")
        
        exe_dir = self.dist_dir / 'SignalAnalyzer'
        if not exe_dir.exists():
            print("ERROR: Executable directory not found!")
            return None
        
        # Create ZIP file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"SignalAnalyzer_v1.0.2_{timestamp}.zip"
        zip_path = self.dist_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in exe_dir.rglob('*'):
                if file_path.is_file():
                    # Add file to zip with relative path
                    arcname = file_path.relative_to(exe_dir)
                    zipf.write(file_path, arcname)
        
        print(f"SUCCESS: Package created: {zip_path}")
        print(f"   Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
        
        return zip_path

    def test_executable(self):
        """Test if the executable runs"""
        print(">> Testing executable...")
        
        exe_path = self.dist_dir / 'SignalAnalyzer' / 'SignalAnalyzer.exe'
        if not exe_path.exists():
            print("ERROR: Executable not found!")
            return False
        
        try:
            # Try to run the executable with a timeout
            process = subprocess.Popen([str(exe_path)], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            # Wait a few seconds to see if it starts
            try:
                stdout, stderr = process.communicate(timeout=5)
                if process.returncode == 0:
                    print("SUCCESS: Executable test passed!")
                    return True
                else:
                    print(f"WARNING: Executable returned code: {process.returncode}")
                    if stderr:
                        print(f"   Error: {stderr.decode()}")
                    return False
            except subprocess.TimeoutExpired:
                # If it's still running after 5 seconds, it probably started successfully
                process.terminate()
                print("SUCCESS: Executable appears to be running successfully!")
                return True
                
        except Exception as e:
            print(f"ERROR: Testing executable failed: {e}")
            return False

    def build(self):
        """Main build process"""
        print(">> Starting Signal Analyzer build process...")
        print("=" * 50)
        
        try:
            # Step 1: Clean previous builds
            self.clean_previous_builds()
            
            # Step 2: Create executable
            if not self.create_executable():
                return False
            
            # Step 3: Copy additional files
            if not self.copy_additional_files():
                return False
            
            # Step 4: Test executable
            if not self.test_executable():
                print("WARNING: Executable test failed, but build may still work")
            
            # Step 5: Create distribution package
            zip_path = self.create_installer_package()
            
            print("=" * 50)
            print("SUCCESS: Build completed successfully!")
            print(f">> Executable location: {self.dist_dir / 'SignalAnalyzer'}")
            if zip_path:
                print(f">> Distribution package: {zip_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Build failed: {e}")
            return False

def main():
    builder = SignalAnalyzerBuilder()
    success = builder.build()
    
    if success:
        print("\nSUCCESS: Build process completed successfully!")
        print("You can now distribute the SignalAnalyzer folder or ZIP file.")
    else:
        print("\nERROR: Build process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()