"""
Windows-Compatible Release Builder for Signal Analyzer
This version uses ASCII characters instead of Unicode emojis for better Windows compatibility
Place this in the root directory as release_builder.py
"""

import sys
import os
import subprocess
import platform
import shutil
import json
import zipfile
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Try to import requests, provide helpful error if missing
try:
    import requests
except ImportError:
    print("ERROR: requests module not found!")
    print("Install with: pip install requests")
    print("Then run: pip install pyinstaller")
    sys.exit(1)

class WindowsCompatibleReleaseBuilder:
    """Windows-compatible release builder with ASCII-only output"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version = self.get_version()
        self.platform_tag = f"{platform.system().lower()}_{platform.machine().lower()}"
        
        # Create release directory structure
        self.releases_base = self.project_root / "releases"
        self.release_dir = self.releases_base / f"v{self.version}_{self.platform_tag}_{self.timestamp}"
        self.release_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with Windows-compatible encoding
        self.setup_logging()
        
        # Build configuration
        self.app_name = "SignalAnalyzer"
        self.build_success = False
        self.test_results = {}
        self.package_paths = []
        
        # GitHub configuration
        self.github_token = None
        self.github_repo = None
        self.github_owner = None
        self.setup_github_config()
        
        # Display banner
        self.display_banner()
        
    def setup_logging(self):
        """Setup Windows-compatible logging"""
        log_file = self.release_dir / f"release_build_{self.timestamp}.log"
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # Create console handler with system encoding
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Release builder logging initialized - {log_file}")
    
    def display_banner(self):
        """Display Windows-compatible banner"""
        banner = f"""
{'='*80}
>> SIGNAL ANALYZER GITHUB RELEASE BUILDER v3.0
{'='*80}
Build Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {self.version}
Platform: {self.platform_tag}
Release Dir: {self.release_dir}
Python: {sys.version.split()[0]}
System: {platform.platform()}
GitHub: {'CONFIGURED' if self.github_token else 'NOT CONFIGURED'}
{'='*80}
"""
        print(banner)
        self.logger.info("GitHub Release builder initialized")
    
    def setup_github_config(self):
        """Setup GitHub configuration from environment or config file"""
        # Try to get GitHub token from environment
        self.github_token = os.environ.get('GITHUB_TOKEN')
        
        # Try to get from config file
        if not self.github_token:
            config_file = self.project_root / ".github_config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        self.github_token = config.get('token')
                        self.github_repo = config.get('repo')
                        self.github_owner = config.get('owner')
                except Exception as e:
                    self.logger.warning(f"Could not load GitHub config: {e}")
        
        # Try to detect repo from git remote
        if not self.github_repo or not self.github_owner:
            try:
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    remote_url = result.stdout.strip()
                    # Parse GitHub URL
                    import re
                    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', remote_url)
                    if match:
                        self.github_owner = match.group(1)
                        self.github_repo = match.group(2)
                        self.logger.info(f"Detected GitHub repo: {self.github_owner}/{self.github_repo}")
            except Exception as e:
                self.logger.debug(f"Could not detect GitHub repo: {e}")
        
        # Create example config file if GitHub not configured
        if not self.github_token:
            self.create_github_config_example()
    
    def create_github_config_example(self):
        """Create an example GitHub configuration file"""
        config_example = {
            "token": "YOUR_GITHUB_TOKEN_HERE",
            "owner": "your-username",
            "repo": "signal-analyzer",
            "instructions": [
                "1. Create a GitHub Personal Access Token:",
                "   - Go to GitHub Settings > Developer settings > Personal access tokens",
                "   - Create a new token with 'repo' permissions",
                "   - Copy the token and replace YOUR_GITHUB_TOKEN_HERE",
                "2. Update owner and repo to match your GitHub repository",
                "3. Rename this file to .github_config.json",
                "4. Alternative: Set GITHUB_TOKEN environment variable"
            ]
        }
        
        example_file = self.project_root / ".github_config.example.json"
        if not example_file.exists():
            with open(example_file, 'w') as f:
                json.dump(config_example, f, indent=2)
            self.logger.info(f"Created GitHub config example: {example_file}")
    
    def get_version(self) -> str:
        """Get version from multiple sources"""
        # Try to get version from setup.py
        try:
            setup_file = self.project_root / "setup.py"
            if setup_file.exists():
                with open(setup_file, 'r') as f:
                    content = f.read()
                    import re
                    match = re.search(r'version=["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        
        # Try to get version from version_info.json
        try:
            version_file = self.project_root / "version_info.json"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    return data.get('version', '1.0.0')
        except Exception:
            pass
        
        # Try to get from git tags
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                tag = result.stdout.strip()
                # Remove 'v' prefix if present
                return tag.lstrip('v')
        except Exception:
            pass
        
        # Fallback to date-based version
        return datetime.now().strftime("%Y.%m.%d")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        print("\n" + "="*60)
        print("CHECKING DEPENDENCIES")
        print("="*60)
        
        # Check critical dependencies
        critical_modules = {
            'numpy': 'Scientific computing',
            'scipy': 'Signal processing', 
            'matplotlib': 'Plotting',
            'pandas': 'Data analysis',
            'pyinstaller': 'Executable creation',
            'tkinter': 'GUI framework',
            'requests': 'GitHub API'
        }
        
        missing_modules = []
        
        for module, description in critical_modules.items():
            try:
                if module == 'tkinter':
                    import tkinter
                    print(f"[OK] {module:<12} - {description}")
                    self.logger.info(f"OK: {module}")
                elif module == 'pyinstaller':
                    # PyInstaller is a command-line tool, not an importable module
                    # Check if the command is available
                    result = subprocess.run(
                        ['pyinstaller', '--version'], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    if result.returncode == 0:
                        print(f"[OK] {module:<12} - {description}")
                        self.logger.info(f"OK: {module}")
                    else:
                        raise subprocess.CalledProcessError(result.returncode, 'pyinstaller')
                else:
                    __import__(module.replace('-', '_'))
                    print(f"[OK] {module:<12} - {description}")
                    self.logger.info(f"OK: {module}")
            except (ImportError, subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                missing_modules.append(module)
                print(f"[MISSING] {module:<12} - {description}")
                self.logger.error(f"MISSING: {module}")
        
        if missing_modules:
            print(f"\nERROR: Missing {len(missing_modules)} required modules!")
            print("\nTo fix this, run:")
            if 'pyinstaller' in missing_modules:
                print("  pip install pyinstaller")
            if 'requests' in missing_modules:
                print("  pip install requests")
            print("  pip install -r requirements.txt")
            
            print("\nIf you get permission errors, try:")
            print("  pip install --user pyinstaller requests")
            print("  Or run as Administrator")
            
            return False
        
        print(f"\n[SUCCESS] All {len(critical_modules)} dependencies found!")
        return True
    
    def step_1_validate_environment(self) -> bool:
        """Step 1: Comprehensive environment validation"""
        print("\n" + "="*60)
        print("STEP 1/5: Environment Validation")
        print("="*60)
        
        try:
            # Check dependencies first
            if not self.check_dependencies():
                return False
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                print(f"ERROR: Python 3.8+ required, found {python_version.major}.{python_version.minor}")
                return False
            print(f"[OK] Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check project structure
            required_paths = [
                self.project_root / "src",
                self.project_root / "src" / "main.py",
                self.project_root / "run.py"
            ]
            
            for path in required_paths:
                if not path.exists():
                    print(f"ERROR: Missing required path: {path}")
                    return False
                print(f"[OK] Found: {path.name}")
            
            # GitHub validation
            if self.github_token:
                if self._validate_github_access():
                    print("[OK] GitHub access validated")
                else:
                    print("[WARNING] GitHub access failed - will skip GitHub operations")
            else:
                print("[INFO] GitHub not configured - will create local packages only")
            
            print("\n[SUCCESS] Environment validation passed!")
            return True
                
        except Exception as e:
            print(f"ERROR: Environment validation failed: {e}")
            return False
    
    def _validate_github_access(self) -> bool:
        """Validate GitHub API access"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Test API access
            response = requests.get('https://api.github.com/user', headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"GitHub API access failed: {response.status_code}")
                return False
            
            user_info = response.json()
            print(f"GitHub user: {user_info.get('login', 'Unknown')}")
            
            # Test repo access if configured
            if self.github_owner and self.github_repo:
                repo_url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}'
                response = requests.get(repo_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    print(f"Repository access confirmed: {self.github_owner}/{self.github_repo}")
                    return True
                else:
                    print(f"Repository access failed: {response.status_code}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"GitHub validation error: {e}")
            return False
    
    def step_2_build_executable(self) -> bool:
        """Step 2: Build the executable"""
        print("\n" + "="*60)
        print("STEP 2/5: Building Executable")
        print("="*60)
        
        try:
            # Check if build.py exists
            build_file = self.project_root / "build.py"
            if not build_file.exists():
                print("ERROR: build.py not found")
                return False
            
            print("Running build.py...")
            
            # Run build script
            result = subprocess.run(
                [sys.executable, "build.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Build completed successfully")
                self.build_success = True
                return True
            else:
                print(f"ERROR: Build failed (code: {result.returncode})")
                if result.stderr:
                    print(f"Error details: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("ERROR: Build timed out")
            return False
        except Exception as e:
            print(f"ERROR: Build failed: {e}")
            return False
    
    def step_3_test_executable(self) -> bool:
        """Step 3: Test the built executable"""
        print("\n" + "="*60)
        print("STEP 3/5: Testing Executable")
        print("="*60)
        
        # Find executable
        dist_dir = self.project_root / "dist"
        if not dist_dir.exists():
            print(f"ERROR: Distribution directory not found: {dist_dir}")
            return False
        
        system = platform.system().lower()
        if system == "windows":
            exe_path = dist_dir / self.app_name / f"{self.app_name}.exe"
        else:
            exe_path = dist_dir / self.app_name / self.app_name
        
        if not exe_path.exists():
            print(f"ERROR: Executable not found: {exe_path}")
            return False
        
        # Test basic execution
        try:
            print("Testing executable startup...")
            
            process = subprocess.Popen(
                [str(exe_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=exe_path.parent
            )
            
            # Wait a few seconds
            import time
            time.sleep(3)
            
            if process.poll() is None:
                # Still running - terminate
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                print("[SUCCESS] Executable test passed")
                return True
            else:
                print(f"ERROR: Executable exited early (code: {process.returncode})")
                return False
                
        except Exception as e:
            print(f"ERROR: Executable test failed: {e}")
            return False
    
    def step_4_create_distribution(self) -> bool:
        """Step 4: Create distribution packages"""
        print("\n" + "="*60)
        print("STEP 4/5: Creating Distribution")
        print("="*60)
        
        try:
            dist_dir = self.project_root / "dist"
            if not dist_dir.exists():
                print("ERROR: Distribution directory not found")
                return False
            
            # Copy application to release directory
            app_source = dist_dir / self.app_name
            if not app_source.exists():
                print(f"ERROR: Application directory not found: {app_source}")
                return False
            
            release_app_dir = self.release_dir / self.app_name
            if release_app_dir.exists():
                shutil.rmtree(release_app_dir)
            
            print(f"Copying application files...")
            shutil.copytree(app_source, release_app_dir)
            
            # Create documentation and scripts
            self.create_release_documentation()
            self.create_launcher_scripts()
            
            # Create distribution packages
            self.package_paths = self.create_distribution_packages()
            
            if self.package_paths:
                print("[SUCCESS] Distribution creation completed")
                return True
            else:
                print("ERROR: Distribution package creation failed")
                return False
                
        except Exception as e:
            print(f"ERROR: Distribution creation failed: {e}")
            return False
    
    def step_5_create_github_release(self) -> bool:
        """Step 5: Create GitHub release and upload packages"""
        print("\n" + "="*60)
        print("STEP 5/5: Creating GitHub Release")
        print("="*60)
        
        if not self.github_token or not self.github_owner or not self.github_repo:
            print("[INFO] GitHub not configured - skipping GitHub release")
            print("       Your packages are ready in the releases/ folder")
            return True
        
        try:
            # Check if tag already exists
            tag_name = f"v{self.version}"
            if self.check_tag_exists(tag_name):
                print(f"WARNING: Tag {tag_name} already exists")
                
                response = input(f"Tag {tag_name} already exists. Options:\n"
                               "1. Create new version with timestamp (recommended)\n"
                               "2. Delete existing tag and recreate\n"
                               "3. Skip GitHub release\n"
                               "Choose (1/2/3): ").strip()
                
                if response == "1":
                    tag_name = f"v{self.version}-{self.timestamp}"
                    print(f"Using new tag: {tag_name}")
                elif response == "2":
                    if self.delete_tag(tag_name):
                        print(f"Deleted existing tag: {tag_name}")
                    else:
                        print("Failed to delete tag")
                        return False
                else:
                    print("Skipping GitHub release")
                    return True
            
            # Create GitHub release
            release_id = self.create_github_release_entry(tag_name)
            if not release_id:
                return False
            
            # Upload packages
            if not self.upload_release_assets(release_id):
                return False
            
            print("[SUCCESS] GitHub release created!")
            print(f"Release URL: https://github.com/{self.github_owner}/{self.github_repo}/releases/tag/{tag_name}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: GitHub release creation failed: {e}")
            return False
    
    def check_tag_exists(self, tag_name: str) -> bool:
        """Check if a Git tag already exists"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/tags/{tag_name}'
            response = requests.get(url, headers=headers, timeout=10)
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.debug(f"Error checking tag: {e}")
            return False
    
    def delete_tag(self, tag_name: str) -> bool:
        """Delete a Git tag"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Delete the tag reference
            url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/tags/{tag_name}'
            response = requests.delete(url, headers=headers, timeout=10)
            
            if response.status_code == 204:
                # Also try to delete the release if it exists
                releases_url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}/releases/tags/{tag_name}'
                release_response = requests.get(releases_url, headers=headers, timeout=10)
                if release_response.status_code == 200:
                    release_data = release_response.json()
                    delete_url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}/releases/{release_data["id"]}'
                    requests.delete(delete_url, headers=headers, timeout=10)
                
                return True
            else:
                print(f"Failed to delete tag: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error deleting tag: {e}")
            return False
    
    def create_github_release_entry(self, tag_name: str) -> Optional[int]:
        """Create a GitHub release"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json',
                'Content-Type': 'application/json'
            }
            
            # Generate release notes
            release_notes = self.generate_release_notes()
            
            release_data = {
                'tag_name': tag_name,
                'target_commitish': 'main',  # or 'master' depending on your default branch
                'name': f'Signal Analyzer {self.version}',
                'body': release_notes,
                'draft': False,
                'prerelease': False
            }
            
            url = f'https://api.github.com/repos/{self.github_owner}/{self.github_repo}/releases'
            response = requests.post(url, headers=headers, json=release_data, timeout=30)
            
            if response.status_code == 201:
                release_info = response.json()
                print(f"GitHub release created: {release_info['html_url']}")
                return release_info['id']
            else:
                print(f"Failed to create release: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error creating GitHub release: {e}")
            return None
    
    def upload_release_assets(self, release_id: int) -> bool:
        """Upload package files to GitHub release"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            upload_success = True
            
            for package_path in self.package_paths:
                if not package_path.exists():
                    print(f"WARNING: Package not found: {package_path}")
                    continue
                
                print(f"Uploading: {package_path.name}")
                
                # Upload file
                upload_url = f'https://uploads.github.com/repos/{self.github_owner}/{self.github_repo}/releases/{release_id}/assets'
                
                with open(package_path, 'rb') as f:
                    file_data = f.read()
                
                upload_headers = headers.copy()
                upload_headers['Content-Type'] = 'application/zip'
                
                params = {'name': package_path.name}
                
                response = requests.post(
                    upload_url,
                    headers=upload_headers,
                    data=file_data,
                    params=params,
                    timeout=300  # 5 minutes for large files
                )
                
                if response.status_code == 201:
                    asset_info = response.json()
                    download_url = asset_info['browser_download_url']
                    print(f"[OK] Uploaded: {package_path.name}")
                    print(f"     Download: {download_url}")
                else:
                    print(f"ERROR: Upload failed for {package_path.name}: {response.status_code}")
                    upload_success = False
            
            return upload_success
            
        except Exception as e:
            print(f"ERROR: Error uploading assets: {e}")
            return False
    
    def generate_release_notes(self) -> str:
        """Generate comprehensive release notes"""
        release_notes = f"""# Signal Analyzer v{self.version}

## What's New

Enhanced signal processing capabilities with improved user interface and better cross-platform compatibility.

## Downloads

- **Windows**: Download the `windows` package for Windows 10/11
- **Linux**: Download the `linux` package for most Linux distributions  
- **macOS**: Download the `darwin` package for macOS 10.14+
- **Portable**: Download the `Portable` version for minimal installation

## Installation

1. Download the appropriate package for your operating system
2. Extract ALL files from the ZIP archive
3. Run the launcher script:
   - **Windows**: Double-click `Start_SignalAnalyzer.bat`
   - **Linux/Mac**: Run `./start_signalanalyzer.sh`

## Features

- **Signal Processing**: Advanced filtering with Savitzky-Golay, Butterworth, and Wavelet filters
- **Interactive Analysis**: Real-time parameter adjustment and preview
- **Peak Detection**: Automatic identification and analysis of signal features
- **Data Export**: Multiple export formats including CSV and high-quality plots
- **User-Friendly Interface**: Intuitive tabbed interface with comprehensive help

## System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Disk Space**: 1GB free space
- **Display**: 1024x768 resolution

### Recommended
- **RAM**: 8GB or more for large files
- **Disk Space**: 2GB free space
- **Display**: 1920x1080 or higher

## Build Information

- **Build Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Platform**: {platform.platform()}
- **Python Version**: {sys.version.split()[0]}
- **Build ID**: {self.timestamp}

## Support

If you encounter any issues:

1. Check the included documentation
2. Review the troubleshooting guide
3. Check the logs folder for error details
4. Open an issue on GitHub with system information

---

**Note**: First time running? The application may take 10-30 seconds to start while it initializes.
"""
        return release_notes
    
    def create_release_documentation(self):
        """Create comprehensive release documentation"""
        print("Creating documentation...")
        
        # Create main README
        readme_content = f"""# Signal Analyzer v{self.version}

## Quick Start Guide

### Installation
1. Download the ZIP package for your operating system
2. Extract ALL files to a new folder
3. Run the launcher script for your platform

### Windows
```
Start_SignalAnalyzer.bat
```

### Linux/Mac
```
./start_signalanalyzer.sh
```

## Features
- Advanced signal processing and filtering
- Interactive data visualization
- Peak detection and analysis
- Multiple export formats
- Cross-platform compatibility

## System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 1GB free space
- **Display**: 1024x768 minimum

## Support
- Check the User Guide for detailed instructions
- Review troubleshooting steps for common issues
- See logs folder for diagnostic information

## Version Information
- **Version**: {self.version}
- **Build Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Platform**: {platform.platform()}

For complete documentation, see the included files.
"""
        
        readme_path = self.release_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("Documentation created")
    
    def create_launcher_scripts(self):
        """Create platform-specific launcher scripts"""
        print("Creating launcher scripts...")
        
        system = platform.system().lower()
        
        if system == "windows":
            # Windows batch file
            batch_content = f'''@echo off
title Signal Analyzer v{self.version}
echo Starting Signal Analyzer v{self.version}...
echo Platform: {platform.platform()}
echo.

cd /d "%~dp0"

if not exist "{self.app_name}\\{self.app_name}.exe" (
    echo ERROR: {self.app_name}.exe not found!
    echo Please ensure all files were extracted from the ZIP.
    pause
    exit /b 1
)

echo Starting application...
cd {self.app_name}
start "" "{self.app_name}.exe"

timeout /t 3 /nobreak >nul
echo Application started. You may close this window.
'''
            script_path = self.release_dir / f"Start_{self.app_name}.bat"
            
        else:
            # Unix shell script
            shell_content = f'''#!/bin/bash
echo "Starting Signal Analyzer v{self.version}..."
echo "Platform: {platform.platform()}"
echo

cd "$(dirname "$0")"

if [ ! -f "{self.app_name}/{self.app_name}" ]; then
    echo "ERROR: {self.app_name} executable not found!"
    echo "Please ensure all files were extracted from the ZIP."
    exit 1
fi

chmod +x {self.app_name}/{self.app_name}

echo "Starting application..."
cd {self.app_name}
./{self.app_name} &

sleep 2
echo "Application started."
'''
            script_path = self.release_dir / f"start_{self.app_name.lower()}.sh"
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(batch_content if system == "windows" else shell_content)
            
            if system != "windows":
                os.chmod(script_path, 0o755)
            
            print(f"Created launcher: {script_path.name}")
            
        except Exception as e:
            print(f"WARNING: Could not create launcher: {e}")
    
    def create_distribution_packages(self) -> List[Path]:
        """Create distribution packages"""
        print("Creating distribution packages...")
        
        packages = []
        
        # Main package
        main_package = self.create_main_package()
        if main_package:
            packages.append(main_package)
        
        # Portable package
        portable_package = self.create_portable_package()
        if portable_package:
            packages.append(portable_package)
        
        return packages
    
    def create_main_package(self) -> Optional[Path]:
        """Create the main distribution package"""
        try:
            package_name = f"{self.app_name}_v{self.version}_{self.platform_tag}_{self.timestamp}"
            zip_path = self.releases_base / f"{package_name}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                # Add entire release directory
                for file_path in self.release_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.release_dir)
                        zipf.write(file_path, arcname)
            
            package_size = zip_path.stat().st_size / (1024 * 1024)
            print(f"Main package: {zip_path.name} ({package_size:.1f} MB)")
            return zip_path
            
        except Exception as e:
            print(f"ERROR: Main package creation failed: {e}")
            return None
    
    def create_portable_package(self) -> Optional[Path]:
        """Create portable package"""
        try:
            package_name = f"{self.app_name}_Portable_v{self.version}_{self.platform_tag}_{self.timestamp}"
            zip_path = self.releases_base / f"{package_name}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
                # Add application
                app_dir = self.release_dir / self.app_name
                if app_dir.exists():
                    for file_path in app_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = self.app_name / file_path.relative_to(app_dir)
                            zipf.write(file_path, arcname)
                
                # Add essential files
                essential_files = ["README.md"]
                for filename in essential_files:
                    file_path = self.release_dir / filename
                    if file_path.exists():
                        zipf.write(file_path, filename)
                
                # Add launcher
                for script_file in self.release_dir.glob("Start_*"):
                    if script_file.is_file():
                        zipf.write(script_file, script_file.name)
                for script_file in self.release_dir.glob("start_*"):
                    if script_file.is_file():
                        zipf.write(script_file, script_file.name)
            
            package_size = zip_path.stat().st_size / (1024 * 1024)
            print(f"Portable package: {zip_path.name} ({package_size:.1f} MB)")
            return zip_path
            
        except Exception as e:
            print(f"WARNING: Portable package creation failed: {e}")
            return None
    
    def run_complete_release_build(self) -> bool:
        """Run the complete release build process"""
        steps = [
            ("Environment Validation", self.step_1_validate_environment),
            ("Building Executable", self.step_2_build_executable),
            ("Testing Executable", self.step_3_test_executable),
            ("Creating Distribution", self.step_4_create_distribution),
            ("Creating GitHub Release", self.step_5_create_github_release),
        ]
        
        success_count = 0
        start_time = datetime.now()
        
        try:
            for step_num, (step_name, step_func) in enumerate(steps, 1):
                step_start = datetime.now()
                
                print(f"\n{'='*60}")
                print(f"STEP {step_num}/{len(steps)}: {step_name}")
                print(f"{'='*60}")
                
                try:
                    if step_func():
                        step_duration = (datetime.now() - step_start).total_seconds()
                        print(f"[SUCCESS] {step_name} completed in {step_duration:.1f}s")
                        success_count += 1
                    else:
                        print(f"[FAILED] {step_name}")
                        break
                        
                except Exception as e:
                    print(f"[EXCEPTION] {step_name} - {e}")
                    self.logger.debug(traceback.format_exc())
                    break
            
            # Final results
            total_duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n{'='*80}")
            print("RELEASE BUILD SUMMARY")
            print(f"{'='*80}")
            
            if success_count == len(steps):
                print("[SUCCESS] RELEASE BUILD COMPLETED!")
                print(f"Total time: {total_duration:.1f} seconds")
                
                if self.github_token:
                    print("GitHub release created!")
                    print(f"View at: https://github.com/{self.github_owner}/{self.github_repo}/releases")
                else:
                    print("Local packages created (GitHub not configured)")
                
                # List created packages
                if self.package_paths:
                    print(f"\nPackages Created:")
                    for package in self.package_paths:
                        size_mb = package.stat().st_size / (1024 * 1024)
                        print(f"  {package.name} ({size_mb:.1f} MB)")
                
                return True
            else:
                print(f"[FAILED] Release build failed at step {success_count + 1}/{len(steps)}")
                return False
                
        except KeyboardInterrupt:
            print("\nRelease build interrupted by user")
            return False
        except Exception as e:
            print(f"\nFATAL ERROR in release build: {e}")
            return False

def main():
    """Main entry point"""
    try:
        print(">> Starting Windows-Compatible Release Builder...")
        
        builder = WindowsCompatibleReleaseBuilder()
        success = builder.run_complete_release_build()
        
        if success:
            print(f"\n{'='*80}")
            print("[SUCCESS] Release build completed!")
            print("Your Signal Analyzer is ready for distribution!")
            print(f"{'='*80}")
            return 0
        else:
            print(f"\n{'='*80}")
            print("[FAILED] Release build encountered errors!")
            print("Check the detailed output above")
            print(f"{'='*80}")
            return 1
            
    except KeyboardInterrupt:
        print("\nRelease build interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())