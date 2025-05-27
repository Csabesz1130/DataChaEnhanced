#!/usr/bin/env python3
"""
One-Click Deployment Setup for Signal Analyzer
This script sets up everything needed for automatic building and distribution.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import json
import urllib.request

def print_step(step_num, title, description=""):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    if description:
        print(f"📋 {description}")
    print('='*60)

def run_command(command, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_python_packages():
    """Check and install required Python packages"""
    packages = [
        'pyinstaller>=5.7.0',
        'pillow>=8.0.0',
        'flask>=2.0.0',
        'requests>=2.25.0'
    ]
    
    print("🔍 Checking Python packages...")
    
    missing_packages = []
    for package in packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✅ {package_name} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package_name} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                return False
    
    return True

def create_github_workflow():
    """Create GitHub Actions workflow"""
    workflow_dir = Path('.github/workflows')
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = '''name: Build and Release Signal Analyzer

on:
  push:
    tags:
      - 'v*.*.*'  # Trigger on version tags like v1.0.0
  workflow_dispatch:  # Allow manual triggering
    inputs:
      version:
        description: 'Version number (e.g., 1.0.0)'
        required: true
        default: '1.0.0'

jobs:
  build-windows:
    runs-on: windows-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r build_requirements.txt
        pip install pillow  # For icon creation
        
    - name: Build executable
      run: |
        python build.py
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: signal-analyzer-windows
        path: dist/SignalAnalyzer_*.zip
        
    - name: Create Release
      if: startsWith(github.ref, 'refs/tags/') || github.event_name == 'workflow_dispatch'
      uses: softprops/action-gh-release@v1
      with:
        tag_name: ${{ github.event.inputs.version && format('v{0}', github.event.inputs.version) || github.ref_name }}
        name: Signal Analyzer ${{ github.event.inputs.version || github.ref_name }}
        draft: false
        prerelease: false
        generate_release_notes: true
        files: |
          dist/SignalAnalyzer_*.zip
        body: |
          ## Signal Analyzer Release
          
          ### Download Instructions
          1. Download the `SignalAnalyzer_*.zip` file
          2. Extract the ZIP file to a folder on your computer
          3. Run `SignalAnalyzer.exe` from the extracted folder
          
          ### System Requirements
          - Windows 10 or later
          - No additional software installation required
          
          **For Teachers:** This is the latest stable version for classroom use.
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
'''
    
    workflow_file = workflow_dir / 'build-release.yml'
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    print(f"✅ Created GitHub workflow: {workflow_file}")
    return True

def create_readme():
    """Create or update README with download instructions"""
    readme_content = '''# Signal Analyzer

Advanced Signal Processing Tool for Education

## 📥 Download Latest Version

**For Teachers and Students:**

Always download the latest version from: [**Latest Release**](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest)

### Installation Instructions:
1. Click the "Latest Release" link above
2. Download the `SignalAnalyzer_*.zip` file
3. Extract the ZIP file to a folder on your computer
4. Double-click `SignalAnalyzer.exe` to run

### System Requirements:
- Windows 10 or later
- 4GB RAM recommended
- No additional software installation required

## 🔄 For Developers

### Building from Source:
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r build_requirements.txt

# Build executable
python build.py
```

### Creating a Release:
```bash
# Tag and push for automatic build
git tag v1.0.0
git push origin v1.0.0
```

## 📚 Features

- Advanced signal filtering and analysis
- Interactive GUI with real-time preview
- Support for various signal formats
- Educational tools for learning signal processing

## 🆘 Support

For technical support, contact your instructor or create an issue in this repository.
'''
    
    readme_file = Path('README.md')
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_file}")
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = '''# Build artifacts
build/
dist/
*.spec

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Config files with secrets
release_config.ini
credentials.json

# Release files
releases/
*.zip
'''
    
    gitignore_file = Path('.gitignore')
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    
    print(f"✅ Created .gitignore: {gitignore_file}")
    return True

def create_version_file():
    """Create version.json file"""
    version_data = {
        "version": "1.0.0",
        "build_date": "2024-01-01",
        "description": "Signal Analyzer - Advanced Signal Processing Tool",
        "author": "Your Name",
        "repository": "https://github.com/YOUR_USERNAME/YOUR_REPO"
    }
    
    version_file = Path('version.json')
    with open(version_file, 'w') as f:
        json.dump(version_data, f, indent=2)
    
    print(f"✅ Created version file: {version_file}")
    return True

def test_build():
    """Test the build process"""
    print("🧪 Testing build process...")
    
    # Check if build.py exists
    if not Path('build.py').exists():
        print("❌ build.py not found! Please ensure the enhanced build.py file is in your project.")
        return False
    
    # Check if run.py exists
    if not Path('run.py').exists():
        print("❌ run.py not found! This is required for building.")
        return False
    
    # Try a test build
    print("🔨 Running test build (this may take a few minutes)...")
    if run_command("python build.py", "Building Signal Analyzer"):
        print("✅ Build test successful!")
        
        # Check if output files exist
        if Path('dist').exists():
            zip_files = list(Path('dist').glob('*.zip'))
            if zip_files:
                print(f"✅ Distribution file created: {zip_files[0]}")
                return True
    
    print("❌ Build test failed. Please check your setup.")
    return False

def setup_git_hooks():
    """Set up Git hooks for easy releases"""
    hooks_dir = Path('.git/hooks')
    if not hooks_dir.exists():
        print("⚠️  Not a Git repository. Initialize Git first with 'git init'")
        return False
    
    # Create a pre-push hook that reminds about tagging
    pre_push_hook = hooks_dir / 'pre-push'
    hook_content = '''#!/bin/sh
echo "🏷️  Remember to create a tag for releases:"
echo "   git tag v1.0.0"
echo "   git push origin v1.0.0"
echo ""
'''
    
    with open(pre_push_hook, 'w') as f:
        f.write(hook_content)
    
    # Make it executable (Unix systems)
    try:
        os.chmod(pre_push_hook, 0o755)
    except:
        pass  # Windows doesn't need this
    
    print("✅ Git hooks set up")
    return True

def generate_instructions():
    """Generate final instructions for the user"""
    instructions = '''
🎉 SETUP COMPLETE! 

📋 NEXT STEPS:

1. UPDATE YOUR REPOSITORY INFO:
   - Edit README.md and replace YOUR_USERNAME/YOUR_REPO with your actual GitHub repo
   - Update version.json with your information

2. PUSH TO GITHUB:
   git add .
   git commit -m "Set up automatic building and deployment"
   git push origin main

3. CREATE YOUR FIRST RELEASE:
   git tag v1.0.0
   git push origin v1.0.0

4. SHARE WITH TEACHERS:
   Give them this link: https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest
   (Replace YOUR_USERNAME/YOUR_REPO with your actual repository)

🚀 AUTOMATIC PROCESS:
- Every time you push a new tag (like v1.0.1), GitHub will automatically:
  ✅ Build your application
  ✅ Create a new release
  ✅ Upload the ZIP file
  ✅ Make it available for download

📞 SUPPORT:
- If you have issues, check the Actions tab in your GitHub repository
- Make sure all your Python files are committed to Git
- Ensure build.py and run.py are in your project root

🎓 FOR EDUCATORS:
- The download link always points to the latest version
- Students don't need to install anything
- The application runs directly from the downloaded folder
'''
    
    print(instructions)
    
    # Save instructions to file
    with open('DEPLOYMENT_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    
    print("💾 Instructions saved to DEPLOYMENT_INSTRUCTIONS.txt")

def main():
    """Main setup process"""
    print("🚀 Signal Analyzer - Deployment Setup")
    print("This will set up automatic building and distribution for your application.")
    
    # Step 1: Check environment
    print_step(1, "Environment Check", "Verifying Python environment and packages")
    if not check_python_packages():
        print("❌ Environment check failed. Please fix the issues above.")
        return False
    
    # Step 2: Create build files
    print_step(2, "Build Configuration", "Setting up build configuration files")
    create_gitignore()
    create_version_file()
    
    # Step 3: Create GitHub Actions
    print_step(3, "GitHub Actions", "Setting up automatic building on GitHub")
    create_github_workflow()
    
    # Step 4: Create documentation
    print_step(4, "Documentation", "Creating README and documentation")
    create_readme()
    
    # Step 5: Set up Git
    print_step(5, "Git Setup", "Configuring Git hooks and helpers")
    setup_git_hooks()
    
    # Step 6: Test build
    print_step(6, "Build Test", "Testing the build process")
    if not test_build():
        print("⚠️  Build test failed, but setup is complete.")
        print("   You may need to fix build issues before deployment works.")
    
    # Step 7: Generate instructions
    print_step(7, "Completion", "Generating final instructions")
    generate_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
        print("📖 Check DEPLOYMENT_INSTRUCTIONS.txt for next steps.")
    else:
        print("\n❌ Setup failed. Please fix the errors above.")
        sys.exit(1)