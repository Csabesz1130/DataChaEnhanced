#!/usr/bin/env python3
"""
Test script to verify PyInstaller is working correctly
Run this before attempting the full release build
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path

def create_test_script():
    """Create a simple test script to build with PyInstaller"""
    test_script_content = '''
import sys
import tkinter as tk
from tkinter import messagebox

def main():
    print("Test application starting...")
    
    # Create a simple window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show success message
    messagebox.showinfo("PyInstaller Test", 
                       "Success! PyInstaller is working correctly.\\n"
                       "This test executable was built successfully.")
    
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    return test_script_content

def test_pyinstaller():
    """Test if PyInstaller can build a simple executable"""
    print("=" * 60)
    print("PYINSTALLER FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        # Check if PyInstaller command is available
        print("Step 1: Checking PyInstaller command...")
        result = subprocess.run(['pyinstaller', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ PyInstaller found: {version}")
        else:
            print("✗ PyInstaller command failed")
            print(f"Error: {result.stderr}")
            return False
        
        # Create temporary directory for test
        print("\nStep 2: Setting up test environment...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test script
            test_script_path = temp_path / "test_app.py"
            with open(test_script_path, 'w') as f:
                f.write(create_test_script())
            print(f"✓ Test script created: {test_script_path}")
            
            # Try to build with PyInstaller
            print("\nStep 3: Building test executable...")
            build_command = [
                'pyinstaller',
                '--onefile',
                '--windowed',
                '--name=TestApp',
                '--distpath=dist_test',
                '--workpath=build_test',
                '--specpath=.',
                str(test_script_path)
            ]
            
            print(f"Running: {' '.join(build_command)}")
            
            build_result = subprocess.run(
                build_command,
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if build_result.returncode == 0:
                print("✓ Build completed successfully!")
                
                # Check if executable was created
                if sys.platform.startswith('win'):
                    exe_path = temp_path / "dist_test" / "TestApp.exe"
                else:
                    exe_path = temp_path / "dist_test" / "TestApp"
                
                if exe_path.exists():
                    exe_size = exe_path.stat().st_size / (1024 * 1024)
                    print(f"✓ Executable created: {exe_path} ({exe_size:.1f} MB)")
                    
                    # Try to run the executable briefly
                    print("\nStep 4: Testing executable...")
                    try:
                        # Start the process
                        test_process = subprocess.Popen(
                            [str(exe_path)],
                            cwd=exe_path.parent,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        # Wait a moment for it to start
                        import time
                        time.sleep(2)
                        
                        # Check if it's still running (good sign)
                        if test_process.poll() is None:
                            print("✓ Executable started successfully!")
                            try:
                                test_process.terminate()
                                test_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                test_process.kill()
                            return True
                        else:
                            print("✗ Executable exited immediately")
                            stdout, stderr = test_process.communicate()
                            if stderr:
                                print(f"Error output: {stderr.decode()}")
                            return False
                            
                    except Exception as e:
                        print(f"✗ Error testing executable: {e}")
                        return False
                else:
                    print("✗ Executable was not created")
                    return False
            else:
                print("✗ Build failed!")
                print(f"Error: {build_result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("✗ PyInstaller build timed out")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def check_dependencies():
    """Check basic dependencies needed for PyInstaller"""
    print("\nDEPENDENCY CHECK")
    print("-" * 30)
    
    dependencies = [
        ('sys', 'Python standard library'),
        ('tkinter', 'GUI framework'),  
        ('subprocess', 'Process management'),
        ('tempfile', 'Temporary files'),
        ('pathlib', 'Path handling')
    ]
    
    all_good = True
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✓ {module:<12} - {description}")
        except ImportError:
            print(f"✗ {module:<12} - {description} (MISSING)")
            all_good = False
    
    return all_good

def main():
    """Main test function"""
    print("PyInstaller Functionality Test")
    print("This will verify that PyInstaller can build executables on your system")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n" + "="*60)
        print("DEPENDENCY CHECK FAILED")
        print("Some required modules are missing. Please check your Python installation.")
        return 1
    
    # Test PyInstaller
    if test_pyinstaller():
        print("\n" + "="*60)
        print("SUCCESS: PyInstaller is working correctly!")
        print("You can proceed with the full release build.")
        print("Run: python release_builder.py")
        return 0
    else:
        print("\n" + "="*60)
        print("FAILED: PyInstaller test failed")
        print()
        print("Troubleshooting steps:")
        print("1. Try reinstalling PyInstaller: pip install --force-reinstall pyinstaller")
        print("2. Check if you're in a virtual environment")
        print("3. Try running as Administrator (Windows)")
        print("4. Check your antivirus software isn't blocking PyInstaller")
        return 1

if __name__ == "__main__":
    sys.exit(main())