"""
Quick setup script to enable easy startup and auto-updates
Integrates the updater into your Signal Analyzer application
"""

import os
import sys
from pathlib import Path
import json


def integrate_updater():
    """Add auto-update check to main.py"""
    
    main_py = Path("src/main.py")
    if not main_py.exists():
        print("❌ Error: src/main.py not found")
        return False
    
    # Read current main.py
    content = main_py.read_text(encoding='utf-8')
    
    # Check if already integrated
    if 'def check_updates_on_startup' in content:
        print("✅ Updater already integrated in main.py")
        return True
    
    # Create the integration code
    integration_code = """

def check_updates_on_startup(root):
    \"\"\"Check for updates after app starts\"\"\"
    try:
        from src.utils.updater import AutoUpdater
        from src.gui.update_dialog import show_update_dialog
        from src.utils.logger import app_logger
        
        # Read current version
        try:
            with open('version_info.json', 'r') as f:
                version_data = json.load(f)
                current_version = version_data.get('version', '1.0.0')
        except:
            current_version = '1.0.0'
        
        # Check for updates
        app_logger.info("Checking for updates...")
        updater = AutoUpdater(current_version, "yourusername/DataChaEnhanced")
        update_info = updater.check_for_updates()
        
        if update_info.get('available'):
            app_logger.info(f"Update available: {update_info['version']}")
            show_update_dialog(root, update_info)
        else:
            app_logger.info("Application is up to date")
            
    except Exception as e:
        # Silent fail - don't disrupt user if update check fails
        app_logger.debug(f"Update check failed: {e}")
"""
    
    # Find the right place to add it (after main() function definition)
    if 'def main():' in content:
        # Add the function before main()
        content = content.replace('def main():', integration_code + '\ndef main():')
        
        # Add the call to check updates in main() after root.mainloop() is set up
        # Find the line with root.mainloop() and add the check before it
        if 'root.mainloop()' in content:
            lines = content.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                if 'root.mainloop()' in line and 'root.after' not in lines[i-1]:
                    # Add update check before mainloop (with 2 second delay)
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + '# Check for updates after 2 seconds')
                    new_lines.append(' ' * indent + 'root.after(2000, lambda: check_updates_on_startup(root))')
                    new_lines.append(' ' * indent)
                new_lines.append(line)
            content = '\n'.join(new_lines)
        
        # Write back
        main_py.write_text(content, encoding='utf-8')
        print("✅ Updater integrated into main.py")
        return True
    else:
        print("⚠️  Could not find main() function in src/main.py")
        print("   You'll need to manually integrate the updater")
        return False


def create_desktop_shortcut_script():
    """Create a script to generate desktop shortcut"""
    
    script = Path("create_desktop_shortcut.py")
    
    content = '''"""
Create a desktop shortcut for Signal Analyzer
Run this script to add a shortcut to your desktop
"""

import os
import sys
from pathlib import Path

def create_shortcut_windows():
    """Create Windows shortcut"""
    try:
        import win32com.client
        
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "Signal Analyzer.lnk"
        
        # Get absolute path to exe
        exe_path = Path(__file__).parent.absolute() / "SignalAnalyzer.exe"
        if not exe_path.exists():
            # Try dist folder
            exe_path = Path(__file__).parent / "dist" / "SignalAnalyzer" / "SignalAnalyzer.exe"
        
        if not exe_path.exists():
            print("❌ Error: SignalAnalyzer.exe not found")
            print(f"   Looking in: {exe_path}")
            return False
        
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(shortcut_path))
        shortcut.TargetPath = str(exe_path)
        shortcut.WorkingDirectory = str(exe_path.parent)
        shortcut.IconLocation = str(exe_path) + ",0"
        shortcut.Description = "Signal Analyzer - Advanced Signal Processing"
        shortcut.save()
        
        print(f"✅ Desktop shortcut created: {shortcut_path}")
        return True
        
    except ImportError:
        print("❌ Error: pywin32 not installed")
        print("   Install with: pip install pywin32")
        return False
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        return False

def create_shortcut_unix():
    """Create Unix desktop entry (Linux)"""
    try:
        desktop_file = Path.home() / ".local" / "share" / "applications" / "signal-analyzer.desktop"
        desktop_file.parent.mkdir(parents=True, exist_ok=True)
        
        exe_path = Path(__file__).parent.absolute() / "SignalAnalyzer"
        icon_path = Path(__file__).parent.absolute() / "assets" / "icon.ico"
        
        content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Signal Analyzer
Comment=Advanced Signal Processing Tool
Exec={exe_path}
Icon={icon_path}
Terminal=false
Categories=Science;Education;
"""
        desktop_file.write_text(content)
        desktop_file.chmod(0o755)
        
        print(f"✅ Desktop entry created: {desktop_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating desktop entry: {e}")
        return False

if __name__ == "__main__":
    print("Creating desktop shortcut...")
    print()
    
    if sys.platform == "win32":
        success = create_shortcut_windows()
    else:
        success = create_shortcut_unix()
    
    if success:
        print()
        print("✅ Success! You can now launch Signal Analyzer from your desktop.")
    else:
        print()
        print("❌ Failed to create shortcut. You may need to create it manually.")
    
    input("\\nPress Enter to exit...")
'''
    
    script.write_text(content, encoding='utf-8')
    print(f"✅ Created: {script}")


def create_build_installer_script():
    """Create script to build the installer"""
    
    script = Path("build_installer.bat" if sys.platform == "win32" else "build_installer.sh")
    
    if sys.platform == "win32":
        content = '''@echo off
echo ========================================
echo Building Signal Analyzer Installer
echo ========================================
echo.

REM Step 1: Build executable with PyInstaller
echo Step 1/3: Building executable...
python build.py
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

REM Step 2: Check if Inno Setup is installed
echo.
echo Step 2/3: Checking for Inno Setup...
set INNO_SETUP="C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe"
if not exist %INNO_SETUP% (
    echo WARNING: Inno Setup not found!
    echo.
    echo Please download and install Inno Setup from:
    echo https://jrsoftware.org/isdl.php
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
)

REM Step 3: Build installer
echo.
echo Step 3/3: Building installer...
%INNO_SETUP% installer_config.iss
if errorlevel 1 (
    echo ERROR: Installer build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ SUCCESS!
echo ========================================
echo.
echo Installer has been created in the 'installers' folder.
echo You can now distribute SignalAnalyzer_Setup_v*.exe
echo.
pause
'''
    else:
        content = '''#!/bin/bash

echo "========================================"
echo "Building Signal Analyzer Package"
echo "========================================"
echo

# Step 1: Build executable
echo "Step 1/2: Building executable..."
python3 build.py
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

# Step 2: Create tarball
echo
echo "Step 2/2: Creating package..."
cd dist
tar -czf SignalAnalyzer_v1.0.2_linux.tar.gz SignalAnalyzer/
if [ $? -ne 0 ]; then
    echo "ERROR: Package creation failed!"
    exit 1
fi

echo
echo "========================================"
echo "✅ SUCCESS!"
echo "========================================"
echo
echo "Package has been created: dist/SignalAnalyzer_v1.0.2_linux.tar.gz"
echo "You can now distribute this package."
echo
'''
    
    script.write_text(content, encoding='utf-8')
    if sys.platform != "win32":
        script.chmod(0o755)
    
    print(f"✅ Created: {script}")


def update_version_info():
    """Ensure version_info.json has correct repo information"""
    
    version_file = Path("version_info.json")
    
    if version_file.exists():
        try:
            with open(version_file, 'r') as f:
                data = json.load(f)
            
            # Add GitHub repo info if not present
            if 'repository' not in data:
                data['repository'] = "yourusername/DataChaEnhanced"
                data['update_url'] = "https://api.github.com/repos/yourusername/DataChaEnhanced/releases/latest"
                
                with open(version_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Updated: {version_file}")
                print("   ⚠️  Remember to change 'yourusername' to your actual GitHub username!")
            else:
                print(f"✅ Version info already configured: {version_file}")
        except Exception as e:
            print(f"⚠️  Could not update version_info.json: {e}")
    else:
        print("⚠️  version_info.json not found - will be created during build")


def create_readme():
    """Create quick start README"""
    
    readme = Path("EASY_STARTUP_SETUP.md")
    
    content = """# Easy Startup & Auto-Update Setup

## What This Setup Does

This setup provides:
1. **Easy Startup**: Creates Windows installer that adds shortcuts
2. **Auto-Updates**: Checks for updates on startup and allows one-click updates
3. **Professional Distribution**: Proper installer for end users

## Quick Start

### Option 1: Build Installer (Recommended)

1. Install Inno Setup (Windows only):
   - Download from: https://jrsoftware.org/isdl.php
   - Install with default settings

2. Run the build script:
   ```bash
   # Windows
   build_installer.bat
   
   # Linux/Mac
   ./build_installer.sh
   ```

3. Distribute the installer:
   - Windows: `installers/SignalAnalyzer_Setup_v*.exe`
   - Linux: `dist/SignalAnalyzer_v*.tar.gz`

### Option 2: Simple Desktop Shortcut

Just want a desktop icon? Run:
```bash
python create_desktop_shortcut.py
```

## Setting Up GitHub Releases for Auto-Update

1. Update your GitHub repo name in:
   - `installer_config.iss` (line 6)
   - `src/utils/updater.py` (line 269)
   - `src/gui/update_dialog.py` (line 137)

2. Create a release on GitHub:
   ```bash
   git tag v1.0.2
   git push origin v1.0.2
   ```

3. Build and upload your distribution:
   - Go to GitHub → Releases → Create New Release
   - Tag: v1.0.2
   - Upload the ZIP/installer file
   - Add release notes

4. Next time you release:
   - Increment version in `version_info.json`
   - Build and upload new version
   - Users will get automatic update notifications!

## Files Created

- `installer_config.iss` - Inno Setup configuration
- `src/utils/updater.py` - Auto-update module
- `src/gui/update_dialog.py` - Update UI
- `create_desktop_shortcut.py` - Creates desktop shortcut
- `build_installer.bat/sh` - Builds installer
- `docs/easy_startup_and_autoupdate_solutions.md` - Full documentation

## Customization

### Change Repository
Edit these files and replace "yourusername/DataChaEnhanced":
- `src/utils/updater.py`
- `src/gui/update_dialog.py`
- `version_info.json`

### Change App Name
Edit `installer_config.iss`:
- Line 5: `AppName=Your App Name`

### Change Icon
Replace `assets/icon.ico` with your icon file.

## Testing

### Test Update Check
```bash
# Change version in src/main.py temporarily to 1.0.0
# Run app - it should show update dialog if newer version exists on GitHub
python run.py
```

### Test Installer
1. Build installer: `build_installer.bat`
2. Run installer on clean system
3. Verify shortcuts created
4. Launch from Start Menu

## Troubleshooting

### "Inno Setup not found"
- Download from https://jrsoftware.org/isdl.php
- Install to default location
- Run `build_installer.bat` again

### "Update check fails"
- Check internet connection
- Verify GitHub repository name
- Check version_info.json format

### "Shortcut creation fails"
- Install pywin32: `pip install pywin32`
- Run as administrator

## Next Steps

1. ✅ Build installer: `build_installer.bat`
2. ✅ Test on clean system
3. ✅ Create GitHub release with your build
4. ✅ Update version and release again
5. ✅ Verify auto-update works

For detailed information, see: `docs/easy_startup_and_autoupdate_solutions.md`
"""
    
    readme.write_text(content, encoding='utf-8')
    print(f"✅ Created: {readme}")


def main():
    """Run all setup steps"""
    
    print("="*60)
    print("Signal Analyzer - Easy Startup & Auto-Update Setup")
    print("="*60)
    print()
    
    steps = [
        ("Creating desktop shortcut script", create_desktop_shortcut_script),
        ("Creating installer build script", create_build_installer_script),
        ("Updating version info", update_version_info),
        ("Creating README", create_readme),
        ("Integrating updater into app", integrate_updater),
    ]
    
    for i, (description, func) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {description}...")
        try:
            func()
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    print()
    print("="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Update 'yourusername' in the files to your GitHub username")
    print("2. Run: build_installer.bat (Windows) or ./build_installer.sh (Linux)")
    print("3. Test the installer on a clean system")
    print("4. Create a GitHub release with your installer")
    print()
    print("See EASY_STARTUP_SETUP.md for detailed instructions.")
    print()


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")

