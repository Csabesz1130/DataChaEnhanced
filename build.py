import os
import sys
import shutil
import subprocess
import platform
import pkg_resources
from pathlib import Path
from PyInstaller.__main__ import run
import zipfile
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EnhancedSignalAnalyzerBuilder:
    """Enhanced builder with comprehensive validation and error handling"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.assets_dir = self.project_root / "assets"
        self.logs_dir = self.project_root / "logs"
        
        # Create logs directory
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # System info
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        
        # Build configuration
        self.version = "1.0.2"
        self.app_name = "SignalAnalyzer"
        
        self.logger.info("="*60)
        self.logger.info("ENHANCED SIGNAL ANALYZER BUILDER")
        self.logger.info("="*60)
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"System: {self.system} - {self.architecture}")

    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"build_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - {log_file}")

    def validate_environment(self) -> bool:
        """Comprehensive environment validation"""
        self.logger.info("🔍 Validating build environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self.logger.error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        self.logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check critical dependencies
        critical_deps = {
            'numpy': '1.21.0',
            'scipy': '1.7.0', 
            'matplotlib': '3.4.0',
            'pandas': '1.3.0',
            'pyinstaller': '5.0.0'
        }
        
        missing_deps = []
        for dep, min_version in critical_deps.items():
            try:
                pkg_resources.require(f"{dep}>={min_version}")
                installed_version = pkg_resources.get_distribution(dep).version
                self.logger.info(f"✅ {dep}: {installed_version}")
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
                missing_deps.append(f"{dep}>={min_version}")
                self.logger.error(f"❌ {dep}: {e}")
        
        # Check optional dependencies
        optional_deps = ['PyWavelets', 'PIL', 'sklearn']
        for dep in optional_deps:
            try:
                pkg_resources.require(dep)
                version = pkg_resources.get_distribution(dep).version
                self.logger.info(f"✅ {dep}: {version} (optional)")
            except pkg_resources.DistributionNotFound:
                self.logger.warning(f"⚠️  {dep}: not found (optional)")
        
        if missing_deps:
            self.logger.error(f"Missing critical dependencies: {missing_deps}")
            self.logger.error("Install with: pip install " + " ".join(missing_deps))
            return False
        
        # Check project structure
        required_paths = [
            self.src_dir,
            self.src_dir / "main.py",
            self.src_dir / "gui",
            self.src_dir / "utils",
            self.project_root / "run.py"
        ]
        
        for path in required_paths:
            if not path.exists():
                self.logger.error(f"❌ Missing required path: {path}")
                return False
            self.logger.debug(f"✅ Found: {path}")
        
        self.logger.info("✅ Environment validation passed")
        return True

    def freeze_dependencies(self):
        """Create comprehensive requirements freeze"""
        self.logger.info("📦 Freezing dependencies...")
        
        try:
            # Get current pip freeze output
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                  capture_output=True, text=True, check=True)
            
            # Write to build_requirements_freeze.txt
            freeze_file = self.project_root / "build_requirements_freeze.txt"
            with open(freeze_file, 'w') as f:
                f.write(f"# Generated on {platform.system()} {platform.release()}\n")
                f.write(f"# Python {sys.version}\n")
                f.write(f"# Build timestamp: {datetime.now().isoformat()}\n\n")
                f.write(result.stdout)
            
            self.logger.info(f"✅ Dependencies frozen to {freeze_file}")
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️ Could not freeze dependencies: {e}")

    def clean_previous_builds(self):
        """Enhanced cleaning of previous build artifacts"""
        self.logger.info("🧹 Cleaning previous builds...")
        
        paths_to_clean = [
            self.build_dir,
            self.dist_dir,
            self.project_root / "SignalAnalyzer.spec",
            self.project_root / "__pycache__"
        ]
        
        # Find all __pycache__ directories recursively
        for pycache in self.project_root.rglob("__pycache__"):
            paths_to_clean.append(pycache)
            
        # Find all .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            paths_to_clean.append(pyc_file)
        
        cleaned_count = 0
        for path in paths_to_clean:
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                    self.logger.debug(f"🗑️ Removed: {path}")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not remove {path}: {e}")
        
        self.logger.info(f"✅ Cleanup completed ({cleaned_count} items removed)")

    def ensure_assets_exist(self):
        """Enhanced asset creation and validation"""
        self.logger.info("🎨 Ensuring assets exist...")
        
        if not self.assets_dir.exists():
            self.assets_dir.mkdir()
            self.logger.info(f"📁 Created: {self.assets_dir}")
        
        # Create or validate icon
        icon_path = self.assets_dir / "icon.ico"
        if not icon_path.exists():
            self.logger.info("🎨 Creating default icon...")
            if self.create_default_icon(icon_path):
                self.logger.info(f"✅ Created default icon: {icon_path}")
            else:
                self.logger.warning("⚠️ Could not create icon, will proceed without")
                return None
        else:
            self.logger.info(f"✅ Using existing icon: {icon_path}")
        
        return icon_path

    def create_default_icon(self, icon_path: Path) -> bool:
        """Create a default icon if PIL is available"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a 64x64 icon with multiple sizes
            sizes = [16, 32, 48, 64]
            images = []
            
            for size in sizes:
                # Create image with gradient background
                img = Image.new('RGBA', (size, size), (70, 130, 180, 255))
                draw = ImageDraw.Draw(img)
                
                # Draw a simple signal wave pattern
                center = size // 2
                for i in range(0, size, 2):
                    y = center + int(center * 0.3 * ((-1) ** (i // 4)))
                    if 0 <= y < size:
                        draw.rectangle([i, y-1, i+1, y+1], fill=(255, 255, 255, 255))
                
                images.append(img)
            
            # Save as ICO with multiple sizes
            images[0].save(icon_path, format='ICO', sizes=[(img.width, img.height) for img in images])
            return True
            
        except ImportError:
            self.logger.warning("PIL not available for icon creation")
            return False
        except Exception as e:
            self.logger.warning(f"Error creating icon: {e}")
            return False

    def get_all_hidden_imports(self):
        """Comprehensive list of hidden imports with system-specific additions"""
        hidden_imports = [
            # GUI Framework
            'tkinter',
            'tkinter.ttk',
            'tkinter.filedialog',
            'tkinter.messagebox',
            
            # Core Scientific Computing
            'numpy',
            'numpy.core',
            'numpy.core._multiarray_umath',
            'numpy.core._multiarray_tests',
            'numpy.linalg',
            'numpy.fft',
            'numpy.random',
            
            # SciPy
            'scipy',
            'scipy.signal',
            'scipy.fft',
            'scipy.integrate',
            'scipy.interpolate',
            'scipy.optimize',
            'scipy.sparse',
            'scipy.sparse.csgraph',
            'scipy.spatial',
            'scipy.special',
            'scipy.stats',
            
            # Matplotlib
            'matplotlib',
            'matplotlib.pyplot',
            'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'matplotlib.backends.backend_agg',
            'matplotlib.backends._backend_agg',
            'matplotlib.figure',
            'matplotlib.font_manager',
            'matplotlib.ft2font',
            'matplotlib.ttconv',
            'matplotlib._path',
            'matplotlib._image',
            'matplotlib._tri',
            'matplotlib._qhull',
            
            # Pandas
            'pandas',
            'pandas.core',
            'pandas.io',
            'pandas.plotting',
            'pandas._libs',
            'pandas._libs.tslibs',
            
            # Wavelets
            'PyWavelets',
            'pywt',
            'pywt._extensions',
            'pywt._extensions._cwt',
            'pywt._extensions._dwt',
            'pywt._extensions._swt',
            
            # Machine Learning (optional)
            'sklearn',
            'sklearn.linear_model',
            
            # Your application modules - CRITICAL
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
            
            # Standard library modules that might be missed
            'pkg_resources',
            'pkg_resources.py2_warn',
            'packaging',
            'packaging.version',
            'packaging.specifiers',
            'packaging.requirements',
            'dateutil',
            'dateutil.tz',
            'pytz',
            'six',
            'certifi',
        ]
        
        # Add system-specific imports
        if self.system == "windows":
            hidden_imports.extend([
                'win32api',
                'win32con',
                'winsound'
            ])
        elif self.system == "darwin":  # macOS
            hidden_imports.extend([
                'Foundation',
                'AppKit'
            ])
        
        return hidden_imports

    def get_system_specific_options(self):
        """Get system-specific PyInstaller options"""
        options = []
        
        if self.system == "windows":
            options.extend([
                "--exclude-module=tkinter.test",
                "--exclude-module=test",
                "--exclude-module=unittest",
                "--exclude-module=pdb",
                "--exclude-module=doctest",
                "--collect-all=numpy",
                "--collect-all=scipy",
            ])
        elif self.system == "darwin":  # macOS
            options.extend([
                "--exclude-module=tkinter.test",
                "--osx-bundle-identifier=com.signalanalyzer.app",
            ])
        elif self.system == "linux":
            options.extend([
                "--exclude-module=tkinter.test",
            ])
            
        return options

    def create_version_info(self):
        """Enhanced version info creation"""
        self.logger.info("📋 Creating version info...")
        
        version_info = {
            "version": self.version,
            "build_date": datetime.now().isoformat(),
            "build_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "description": "Signal Analyzer - Advanced Signal Processing Tool",
            "company": "Signal Analysis Lab",
            "product": "Signal Analyzer",
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "python_version": sys.version
            },
            "dependencies": self.get_dependency_versions()
        }
        
        version_file = self.project_root / "version_info.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2)
        
        self.logger.info(f"✅ Version info created: {version_file}")
        return version_info

    def get_dependency_versions(self):
        """Get versions of key dependencies"""
        deps = {}
        key_packages = ['numpy', 'scipy', 'matplotlib', 'pandas', 'pyinstaller']
        
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                deps[package] = version
            except pkg_resources.DistributionNotFound:
                deps[package] = "not found"
        
        return deps

    def create_executable(self):
        """Enhanced executable creation with comprehensive error handling"""
        self.logger.info("🔨 Creating executable...")
        
        # Verify run.py exists
        run_script = self.project_root / "run.py"
        if not run_script.exists():
            self.logger.error(f"❌ run.py not found at {run_script}")
            return False
        
        # Get icon path
        icon_path = self.ensure_assets_exist()
        
        # Create version info
        version_info = self.create_version_info()
        
        # Build PyInstaller arguments
        args = [
            str(run_script),
            f"--name={self.app_name}",
            "--onedir",  # Create a folder distribution for better compatibility
            "--windowed",  # No console window
            "--clean",  # Clean cache
            "--noconfirm",  # Don't ask for confirmation
            f"--distpath={self.dist_dir}",
            f"--workpath={self.build_dir}",
            f"--specpath={self.project_root}",
            
            # Critical paths
            f"--paths={self.src_dir}",
            f"--paths={self.project_root}",
            
            # Runtime options for better stability
            "--noupx",  # Don't use UPX compression (can cause issues)
            "--strip",  # Strip debug symbols to reduce size
        ]
        
        # Add icon if available
        if icon_path and icon_path.exists():
            args.append(f"--icon={icon_path}")
            self.logger.info(f"🎨 Using icon: {icon_path}")
        
        # Add source directory as data - CRITICAL
        if self.src_dir.exists():
            if self.system == "windows":
                args.append(f"--add-data={self.src_dir};src")
            else:
                args.append(f"--add-data={self.src_dir}:src")
            self.logger.info(f"📁 Added source directory: {self.src_dir}")
        
        # Add assets directory
        if self.assets_dir.exists():
            if self.system == "windows":
                args.append(f"--add-data={self.assets_dir};assets")
            else:
                args.append(f"--add-data={self.assets_dir}:assets")
            self.logger.info(f"🎨 Added assets directory: {self.assets_dir}")
        
        # Add all hidden imports
        hidden_imports = self.get_all_hidden_imports()
        for import_name in hidden_imports:
            args.append(f"--hidden-import={import_name}")
        self.logger.info(f"📦 Added {len(hidden_imports)} hidden imports")
        
        # Add system-specific options
        system_options = self.get_system_specific_options()
        args.extend(system_options)
        if system_options:
            self.logger.info(f"⚙️ Added {len(system_options)} system-specific options")
        
        # Additional exclusions to reduce size
        exclusions = [
            "PIL.ImageQt",
            "PyQt5",
            "PyQt6",
            "PySide2",
            "PySide6",
            "tkinter.test",
            "test",
            "unittest",
            "pdb",
            "doctest"
        ]
        for exclusion in exclusions:
            args.append(f"--exclude-module={exclusion}")
        
        # Critical collection directives
        args.extend([
            "--collect-all=src",  # Collect all submodules from src
            "--collect-submodules=src",
        ])
        
        self.logger.info(f"🚀 Running PyInstaller with {len(args)} arguments...")
        self.logger.debug(f"PyInstaller command: {' '.join(args)}")
        
        try:
            # Run PyInstaller
            run(args)
            
            # Verify the executable was created
            if self.system == "windows":
                exe_path = self.dist_dir / self.app_name / f"{self.app_name}.exe"
            else:
                exe_path = self.dist_dir / self.app_name / self.app_name
            
            if exe_path.exists():
                exe_size = exe_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"✅ Executable created successfully!")
                self.logger.info(f"📁 Location: {exe_path}")
                self.logger.info(f"📏 Size: {exe_size:.1f} MB")
                return True
            else:
                self.logger.error(f"❌ Executable not found at expected location: {exe_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ PyInstaller failed: {e}")
            # Log more details about the failure
            self.logger.error("This could be due to:")
            self.logger.error("  - Missing dependencies")
            self.logger.error("  - Import errors in your code")
            self.logger.error("  - Path issues")
            self.logger.error("  - Insufficient disk space")
            return False

    def copy_additional_files(self):
        """Enhanced copying of additional files with validation"""
        self.logger.info("📂 Copying additional files...")
        
        exe_dir = self.dist_dir / self.app_name
        if not exe_dir.exists():
            self.logger.error(f"❌ Executable directory not found: {exe_dir}")
            return False
        
        copied_files = 0
        
        # Copy version info
        version_file = self.project_root / "version_info.json"
        if version_file.exists():
            try:
                shutil.copy2(version_file, exe_dir)
                self.logger.info(f"✅ Copied: {version_file.name}")
                copied_files += 1
            except Exception as e:
                self.logger.warning(f"⚠️ Could not copy {version_file.name}: {e}")
        
        # Copy README files
        readme_files = ['README.md', 'README.txt', 'readme.txt']
        for readme in readme_files:
            readme_path = self.project_root / readme
            if readme_path.exists():
                try:
                    shutil.copy2(readme_path, exe_dir)
                    self.logger.info(f"✅ Copied: {readme}")
                    copied_files += 1
                    break
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not copy {readme}: {e}")
        
        # Copy sample data if exists
        data_dir = self.project_root / "data"
        if data_dir.exists():
            sample_files = list(data_dir.glob("*sample*.atf")) + list(data_dir.glob("*example*.atf"))
            if sample_files:
                exe_data_dir = exe_dir / "data"
                exe_data_dir.mkdir(exist_ok=True)
                for sample_file in sample_files[:3]:  # Limit to 3 sample files
                    try:
                        shutil.copy2(sample_file, exe_data_dir)
                        self.logger.info(f"✅ Copied sample: {sample_file.name}")
                        copied_files += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not copy sample {sample_file.name}: {e}")
        
        # Create comprehensive user guide
        self.create_user_guide(exe_dir)
        copied_files += 1
        
        # Create startup script for better error handling
        self.create_startup_script(exe_dir)
        copied_files += 1
        
        self.logger.info(f"✅ Copied {copied_files} additional files")
        return True

    def create_user_guide(self, exe_dir: Path):
        """Create a comprehensive user guide"""
        user_guide = exe_dir / "User_Guide.txt"
        
        guide_content = f"""Signal Analyzer v{self.version} - User Guide
{'='*50}

GETTING STARTED
---------------
1. Double-click {self.app_name}.exe to start the application
2. If you see errors, try running Start_{self.app_name}.bat instead
3. Use File > Load to load your ATF data files
4. Apply filters using the Filter tab
5. Analyze signals using the Analysis tab
6. Export results when done

SYSTEM REQUIREMENTS
-------------------
- {platform.system()} {platform.release()}
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- Display resolution: 1024x768 minimum

SUPPORTED FILE FORMATS
----------------------
- ATF (Axon Text Format) - Primary format
- CSV - Basic support for export

TROUBLESHOOTING
---------------
If the application doesn't start:
1. Check that all files were extracted from the ZIP
2. Try running as Administrator (Windows) or with sudo (Linux/Mac)
3. Check available disk space and memory
4. Temporarily disable antivirus software
5. Check the logs folder for error details

If you see import errors:
1. Make sure you extracted ALL files from the ZIP
2. Don't move individual files - keep the folder structure
3. Try re-downloading and extracting the ZIP file

For performance issues:
1. Close other applications to free memory
2. Use smaller data files for testing
3. Check available disk space

FEATURES
--------
- Interactive signal visualization
- Multiple filtering options (Savitzky-Golay, Butterworth, Wavelet)
- Peak detection and analysis
- Event detection and characterization
- Statistical analysis
- Export capabilities

BUILD INFORMATION
-----------------
Version: {self.version}
Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: {platform.platform()}
Python Version: {sys.version.split()[0]}

For technical support, contact your instructor or check the documentation.
"""
        
        try:
            with open(user_guide, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            self.logger.info(f"✅ Created user guide: {user_guide.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create user guide: {e}")

    def create_startup_script(self, exe_dir: Path):
        """Create a startup script with better error handling"""
        if self.system == "windows":
            script_content = f'''@echo off
title Signal Analyzer v{self.version}
echo Starting Signal Analyzer v{self.version}...
echo Platform: {platform.platform()}
echo.

cd /d "%~dp0"

echo Checking files...
if not exist "{self.app_name}.exe" (
    echo ERROR: {self.app_name}.exe not found!
    echo Please ensure all files were extracted correctly.
    echo Check that you extracted the entire ZIP file.
    pause
    exit /b 1
)

echo Files OK. Starting application...
{self.app_name}.exe

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Signal Analyzer failed to start (Exit code: %ERRORLEVEL%)
    echo.
    echo Troubleshooting steps:
    echo 1. Run as Administrator
    echo 2. Check antivirus software settings
    echo 3. Ensure all files are extracted
    echo 4. Check available disk space and memory
    echo 5. Check logs folder for details
    echo.
    pause
)
'''
            script_path = exe_dir.parent / f"Start_{self.app_name}.bat"
        else:
            script_content = f'''#!/bin/bash

echo "Starting Signal Analyzer v{self.version}..."
echo "Platform: {platform.platform()}"
echo

cd "$(dirname "$0")/{self.app_name}"

echo "Checking files..."
if [ ! -f "{self.app_name}" ]; then
    echo "ERROR: {self.app_name} executable not found!"
    echo "Please ensure all files were extracted correctly."
    echo "Check that you extracted the entire ZIP file."
    exit 1
fi

# Make executable if needed
chmod +x {self.app_name}

echo "Files OK. Starting application..."
./{self.app_name}

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Signal Analyzer failed to start"
    echo
    echo "Troubleshooting steps:"
    echo "1. Check file permissions"
    echo "2. Install required system libraries"
    echo "3. Check available disk space and memory"
    echo "4. Check logs folder for details"
    echo
fi
'''
            script_path = exe_dir.parent / f"start_{self.app_name.lower()}.sh"
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            if self.system != "windows":
                os.chmod(script_path, 0o755)
            
            self.logger.info(f"✅ Created startup script: {script_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create startup script: {e}")

    def test_executable(self):
        """Enhanced executable testing with detailed diagnostics"""
        self.logger.info("🧪 Testing executable...")
        
        if self.system == "windows":
            exe_path = self.dist_dir / self.app_name / f"{self.app_name}.exe"
        else:
            exe_path = self.dist_dir / self.app_name / self.app_name
        
        if not exe_path.exists():
            self.logger.error(f"❌ Executable not found: {exe_path}")
            return False
        
        # Check file permissions
        if not os.access(exe_path, os.X_OK):
            self.logger.warning(f"⚠️ Executable lacks execute permission: {exe_path}")
            if self.system != "windows":
                try:
                    os.chmod(exe_path, 0o755)
                    self.logger.info("✅ Fixed execute permissions")
                except Exception as e:
                    self.logger.error(f"❌ Could not fix permissions: {e}")
                    return False
        
        # Test basic execution
        try:
            self.logger.info("🚀 Starting executable test...")
            
            # Create a simple test environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(exe_path.parent)
            
            process = subprocess.Popen(
                [str(exe_path)], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=exe_path.parent,
                env=env
            )
            
            # Wait a few seconds to see if it starts properly
            import time
            time.sleep(3)
            
            if process.poll() is None:
                # Still running - try to terminate gracefully
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    self.logger.info("✅ Executable started and terminated successfully!")
                    return True
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.logger.info("✅ Executable started successfully (force killed)!")
                    return True
            else:
                # Process ended quickly - check return code
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    self.logger.info("✅ Executable test completed successfully!")
                    return True
                else:
                    self.logger.error(f"❌ Executable failed with code: {process.returncode}")
                    if stderr:
                        stderr_text = stderr.decode('utf-8', errors='ignore')
                        self.logger.error(f"Error output: {stderr_text[:500]}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Executable test failed: {e}")
            return False

    def create_installer_package(self):
        """Enhanced package creation with validation"""
        self.logger.info("📦 Creating distribution package...")
        
        exe_dir = self.dist_dir / self.app_name
        if not exe_dir.exists():
            self.logger.error(f"❌ Executable directory not found: {exe_dir}")
            return None
        
        # Create package info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{self.app_name}_v{self.version}_{self.system}_{timestamp}.zip"
        zip_path = self.dist_dir / zip_name
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                file_count = 0
                
                # Add executable directory
                for file_path in exe_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = self.app_name / file_path.relative_to(exe_dir)
                        zipf.write(file_path, arcname)
                        file_count += 1
                
                # Add startup scripts
                for script_file in exe_dir.parent.glob("Start_*"):
                    if script_file.is_file():
                        zipf.write(script_file, script_file.name)
                        file_count += 1
                
                for script_file in exe_dir.parent.glob("start_*"):
                    if script_file.is_file():
                        zipf.write(script_file, script_file.name)
                        file_count += 1
            
            # Validate package
            package_size = zip_path.stat().st_size / (1024 * 1024)
            
            # Test ZIP integrity
            try:
                with zipfile.ZipFile(zip_path, 'r') as test_zip:
                    test_result = test_zip.testzip()
                    if test_result:
                        self.logger.error(f"❌ ZIP integrity check failed: {test_result}")
                        return None
            except Exception as e:
                self.logger.error(f"❌ ZIP validation failed: {e}")
                return None
            
            self.logger.info(f"✅ Package created successfully!")
            self.logger.info(f"📁 Location: {zip_path}")
            self.logger.info(f"📏 Size: {package_size:.1f} MB")
            self.logger.info(f"📄 Files: {file_count}")
            
            return zip_path
            
        except Exception as e:
            self.logger.error(f"❌ Package creation failed: {e}")
            return None

    def generate_build_report(self):
        """Generate a comprehensive build report"""
        self.logger.info("📋 Generating build report...")
        
        report = {
            "build_info": {
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "builder": "EnhancedSignalAnalyzerBuilder",
                "platform": platform.platform(),
                "python_version": sys.version
            },
            "project_info": {
                "project_root": str(self.project_root),
                "source_files": len(list(self.src_dir.rglob("*.py"))) if self.src_dir.exists() else 0,
                "total_size_mb": sum(f.stat().st_size for f in self.project_root.rglob('*') if f.is_file()) / (1024*1024)
            },
            "dependencies": self.get_dependency_versions(),
            "build_artifacts": []
        }
        
        # Check for build artifacts
        if self.dist_dir.exists():
            for item in self.dist_dir.iterdir():
                if item.is_file() and item.suffix == '.zip':
                    report["build_artifacts"].append({
                        "name": item.name,
                        "size_mb": item.stat().st_size / (1024*1024),
                        "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
        
        # Save report
        report_file = self.project_root / "build_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"✅ Build report saved: {report_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save build report: {e}")

    def build(self):
        """Main enhanced build process with comprehensive error handling"""
        self.logger.info("🚀 Starting enhanced Signal Analyzer build process...")
        self.logger.info("="*60)
        
        build_steps = [
            ("Environment Validation", self.validate_environment),
            ("Dependency Freezing", self.freeze_dependencies),
            ("Cleaning Previous Builds", self.clean_previous_builds),
            ("Creating Executable", self.create_executable),
            ("Copying Additional Files", self.copy_additional_files),
            ("Testing Executable", self.test_executable),
            ("Creating Package", self.create_installer_package),
        ]
        
        success_count = 0
        package_path = None
        
        try:
            for step_name, step_func in build_steps:
                self.logger.info(f"\n🔄 Step {success_count + 1}/{len(build_steps)}: {step_name}")
                self.logger.info("-" * 50)
                
                if step_name == "Dependency Freezing":
                    step_func()  # This step doesn't return boolean
                    success_count += 1
                elif step_name == "Creating Package":
                    package_path = step_func()
                    if package_path:
                        success_count += 1
                    else:
                        break
                else:
                    if step_func():
                        success_count += 1
                    else:
                        self.logger.error(f"❌ Step failed: {step_name}")
                        break
            
            # Generate build report
            self.generate_build_report()
            
            # Final summary
            self.logger.info("\n" + "="*60)
            self.logger.info("BUILD SUMMARY")
            self.logger.info("="*60)
            
            if success_count == len(build_steps):
                self.logger.info("🎉 BUILD COMPLETED SUCCESSFULLY!")
                self.logger.info(f"📁 Executable: {self.dist_dir / self.app_name}")
                if package_path:
                    self.logger.info(f"📦 Package: {package_path}")
                    self.logger.info(f"📏 Package size: {package_path.stat().st_size / (1024*1024):.1f} MB")
                
                self.logger.info("\n📋 Next steps:")
                self.logger.info("1. Test the executable on a clean system")
                self.logger.info("2. Distribute the ZIP package")
                self.logger.info("3. Provide the User_Guide.txt to users")
                
                return True
            else:
                self.logger.error(f"❌ BUILD FAILED at step {success_count + 1}/{len(build_steps)}")
                self.logger.error(f"📁 Partial build available in: {self.dist_dir}")
                
                self.logger.info("\n🔧 Troubleshooting tips:")
                self.logger.info("1. Check the full log above for specific errors")
                self.logger.info("2. Ensure all dependencies are installed")
                self.logger.info("3. Verify your source code has no import errors")
                self.logger.info("4. Check available disk space")
                
                return False
                
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️ Build interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"\n💥 Unexpected error during build: {e}")
            self.logger.error("Full traceback:", exc_info=True)
            return False

def main():
    """Main entry point with proper error handling"""
    try:
        builder = EnhancedSignalAnalyzerBuilder()
        success = builder.build()
        
        if success:
            print("\n" + "="*60)
            print("🎉 SUCCESS: Enhanced build process completed!")
            print("✅ Your Signal Analyzer is ready for distribution!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("❌ FAILED: Build process encountered errors!")
            print("📋 Check the log output above for details")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())