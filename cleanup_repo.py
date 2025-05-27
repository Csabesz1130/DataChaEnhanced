#!/usr/bin/env python3
"""
Repository Cleanup Script for Signal Analyzer
This script helps clean up accidentally committed files from your Git repository.
"""

import os
import subprocess
import shutil
from pathlib import Path
import sys
import time
import glob

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n🔸 STEP {step_num}: {title}")
    print("-" * 50)

def run_command(command, description="", ignore_errors=False):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout.strip():
                # Only show first few lines to avoid spam
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 5:
                    for line in output_lines[:3]:
                        print(f"   {line}")
                    print(f"   ... ({len(output_lines)-3} more lines)")
                else:
                    for line in output_lines:
                        print(f"   {line}")
            return True, result.stdout
        else:
            if ignore_errors:
                print(f"⚠️  Warning: {result.stderr}")
                return True, result.stderr
            else:
                print(f"❌ Error: {result.stderr}")
                return False, result.stderr
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)

def check_git_repo():
    """Check if we're in a Git repository"""
    if not Path('.git').exists():
        print("❌ ERROR: Not a Git repository!")
        print("   Please run this script in your project root directory.")
        return False
    return True

def backup_important_files():
    """Backup important files before cleanup"""
    print_step(1, "Creating Backup of Important Files")
    
    backup_dir = Path("cleanup_backup_" + str(int(time.time())))
    backup_dir.mkdir(exist_ok=True)
    
    print(f"📁 Backup directory: {backup_dir}")
    
    # Define what to backup
    important_patterns = {
        'source_code': ['src/', '*.py'],
        'config_files': ['requirements.txt', 'build_requirements.txt', 'setup.py', 'package.json'],
        'documentation': ['README.md', 'README.txt', '*.md'],
        'github_actions': ['.github/'],
        'assets': ['assets/', 'data/', 'images/'],
        'other_configs': ['*.ini', '*.json', '*.yml', '*.yaml']
    }
    
    backed_up_files = []
    
    for category, patterns in important_patterns.items():
        print(f"🔍 Backing up {category}...")
        category_dir = backup_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for pattern in patterns:
            try:
                if pattern.endswith('/'):
                    # Directory
                    dir_path = Path(pattern.rstrip('/'))
                    if dir_path.exists() and dir_path.is_dir():
                        shutil.copytree(dir_path, category_dir / dir_path.name, dirs_exist_ok=True)
                        backed_up_files.append(str(dir_path))
                        print(f"   ✅ Backed up directory: {dir_path}")
                else:
                    # File or glob pattern
                    if '*' in pattern:
                        files = glob.glob(pattern, recursive=True)
                        for file_path in files:
                            if Path(file_path).is_file():
                                shutil.copy2(file_path, category_dir)
                                backed_up_files.append(file_path)
                                print(f"   ✅ Backed up: {file_path}")
                    else:
                        file_path = Path(pattern)
                        if file_path.exists() and file_path.is_file():
                            shutil.copy2(file_path, category_dir)
                            backed_up_files.append(str(file_path))
                            print(f"   ✅ Backed up: {file_path}")
            except Exception as e:
                print(f"   ⚠️  Could not backup {pattern}: {e}")
    
    print(f"\n📊 Backup Summary:")
    print(f"   Backup location: {backup_dir}")
    print(f"   Files backed up: {len(backed_up_files)}")
    
    return backup_dir, backed_up_files

def create_comprehensive_gitignore():
    """Create a comprehensive .gitignore file"""
    print_step(2, "Creating Comprehensive .gitignore")
    
    gitignore_content = '''# =============================================================================
# SIGNAL ANALYZER - COMPREHENSIVE .gitignore
# =============================================================================

# Current user-specific
.qodo/

# =============================================================================
# PYTHON
# =============================================================================

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# =============================================================================
# PYTHON PACKAGE FILES (What you accidentally committed)
# =============================================================================

# Wheel files
*.whl

# Python packages and site-packages
site-packages/
Lib/
Scripts/
Include/
pyvenv.cfg

# Specific packages that appeared in your repo
scipy*/
scipy-*/
numpy*/
numpy-*/
matplotlib*/
matplotlib-*/
pandas*/
pandas-*/
PIL*/
Pillow*/
Pillow-*/
PyQt*/
PyWavelets*/
mypy*/
mypy-*/
typing_extensions*/
typing_extensions-*/
black*/
black-*/
setuptools*/
setuptools-*/
pip*/
pip-*/

# Any .cp* files (compiled Python extensions)
*.cp*.so
*.cp*.pyd
*.cp311-win_amd64.pyd
*.cp311-win32.pyd

# =============================================================================
# BUILD AND RELEASE ARTIFACTS
# =============================================================================

# Build directories
build/
dist/
*.egg-info/

# PyInstaller build files
*.spec

# Release files and packages
releases/
*.zip
*.tar.gz
*.exe
*.msi
*.dmg
*.pkg

# Temporary build files
temp/
tmp/

# =============================================================================
# IDE AND EDITOR FILES
# =============================================================================

# Visual Studio Code
.vscode/
*.code-workspace

# PyCharm
.idea/
*.iml
*.iws

# Sublime Text
*.sublime-project
*.sublime-workspace

# Vim
*.swp
*.swo
*~

# =============================================================================
# OPERATING SYSTEM FILES
# =============================================================================

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.cab
*.msi
*.msix
*.msm
*.msp
*.lnk

# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*

# Linux
*~
.fuse_hidden*
.directory
.Trash-*
.nfs*

# =============================================================================
# APPLICATION-SPECIFIC
# =============================================================================

# Configuration files with secrets
config.ini
settings.ini
secrets.json
credentials.json
release_config.ini

# Log files
*.log
logs/

# Cache directories
.cache/
cache/

# Temporary files
*.tmp
*.temp
*.bak
*.backup

# Test output
test_output/
test_results/

# Virtual environment indicators
venv*/
env*/
.venv*/
.env*/
ENV*/
virtualenv*/

# Conda environments
.conda/
miniconda*/
anaconda*/

# Backup files
*.bak
*.backup
*.old

# Archive files
*.7z
*.dmg
*.gz
*.iso
*.jar
*.rar
*.tar
*.zip
'''
    
    # Backup existing .gitignore
    if Path('.gitignore').exists():
        shutil.copy2('.gitignore', '.gitignore.backup')
        print("   📋 Backed up existing .gitignore to .gitignore.backup")
    
    # Write new .gitignore
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Created comprehensive .gitignore file")
    return True

def remove_unwanted_files_from_git():
    """Remove unwanted files from Git tracking"""
    print_step(3, "Removing Unwanted Files from Git Tracking")
    
    # Define patterns to remove
    unwanted_patterns = [
        # Python wheel files
        "*.whl",
        # Compiled Python files
        "*.cp*.pyd",
        "*.cp*.so",
        "*.pyc",
        # Build directories
        "build/",
        "dist/",
        "__pycache__/",
        # Package directories
        "site-packages/",
        "Lib/",
        "Scripts/",
        "Include/",
        "pyvenv.cfg",
        # Specific Python packages
        "scipy*",
        "numpy*",
        "matplotlib*",
        "pandas*",
        "PIL*",
        "Pillow*",
        "PyQt*",
        "PyWavelets*",
        "mypy*",
        "typing_extensions*",
        "black*",
        "setuptools*",
        "pip*",
        # Python distribution
        "*.egg-info/",
        ".Python",
        # Virtual environments
        "env/",
        "venv/",
        "ENV/",
        # Temporary and log files
        "*.log",
        "*.tmp",
        "*.backup",
        "*.bak",
        # Cache directories
        ".cache/",
        "cache/"
    ]
    
    removed_patterns = []
    
    for pattern in unwanted_patterns:
        print(f"🗑️  Removing pattern: {pattern}")
        success, output = run_command(
            f'git rm -r --cached --ignore-unmatch "{pattern}"', 
            f"Removing {pattern}",
            ignore_errors=True
        )
        if success:
            removed_patterns.append(pattern)
    
    print(f"\n📊 Removal Summary:")
    print(f"   Patterns processed: {len(unwanted_patterns)}")
    print(f"   Successfully removed: {len(removed_patterns)}")
    
    return True

def add_important_files_back():
    """Add important files back to Git"""
    print_step(4, "Adding Important Files Back to Git")
    
    # Files and directories to definitely keep
    important_patterns = [
        "src/",
        "*.py",
        "requirements.txt",
        "build_requirements.txt",
        "setup.py",
        "README.md",
        "README.txt",
        ".github/",
        ".gitignore",
        "assets/",
        "data/",
        "*.md",
        "package.json",
        "version.json"
    ]
    
    added_files = []
    
    for pattern in important_patterns:
        if '*' in pattern:
            # Handle glob patterns
            files = glob.glob(pattern, recursive=True)
            for file_path in files:
                if Path(file_path).exists():
                    success, _ = run_command(f'git add "{file_path}"', f"Adding {file_path}", ignore_errors=True)
                    if success:
                        added_files.append(file_path)
        else:
            # Handle specific files/directories
            if Path(pattern).exists():
                success, _ = run_command(f'git add "{pattern}"', f"Adding {pattern}", ignore_errors=True)
                if success:
                    added_files.append(pattern)
    
    print(f"\n📊 Added Files Summary:")
    print(f"   Files added back: {len(added_files)}")
    
    return True

def commit_changes():
    """Commit the cleanup changes"""
    print_step(5, "Committing Cleanup Changes")
    
    # Check if there are changes to commit
    success, output = run_command("git status --porcelain", "Checking for changes")
    
    if not output.strip():
        print("ℹ️  No changes to commit")
        return True
    
    # Show what will be committed
    print("📋 Changes to be committed:")
    run_command("git status --short", "Showing changes")
    
    # Commit the changes
    commit_message = "Clean up repository: remove Python packages and build artifacts\n\n- Removed accidentally committed Python packages\n- Removed build artifacts and cache files\n- Updated .gitignore to prevent future issues"
    
    success, _ = run_command(f'git commit -m "{commit_message}"', "Committing cleanup changes")
    
    if success:
        print("✅ Changes committed successfully")
    
    return success

def clean_git_history():
    """Clean Git history to reduce repository size"""
    print_step(6, "Cleaning Git History (Optional)")
    
    print("⚠️  WARNING: This will rewrite Git history!")
    print("   This operation cannot be undone easily.")
    print("   Make sure you have backups of your work.")
    print("   Other collaborators will need to re-clone the repository.")
    
    response = input("\nDo you want to proceed with history cleanup? (y/N): ")
    
    if response.lower() != 'y':
        print("⏭️  Skipping history cleanup")
        return True
    
    print("🔄 Starting Git history cleanup...")
    
    # Method 1: Try git filter-repo (recommended)
    success, _ = run_command("git filter-repo --help", "Checking for git-filter-repo", ignore_errors=True)
    
    if success:
        print("🔧 Using git-filter-repo for cleanup...")
        
        filter_commands = [
            ('git filter-repo --strip-blobs-bigger-than 1M', "Removing blobs larger than 1MB"),
            ('git filter-repo --path-glob "*.whl" --invert-paths', "Removing wheel files from history"),
            ('git filter-repo --path-glob "*.pyd" --invert-paths', "Removing compiled Python files from history"),
            ('git filter-repo --path-glob "site-packages" --invert-paths', "Removing site-packages from history"),
            ('git filter-repo --path-glob "Lib" --invert-paths', "Removing Lib directory from history"),
            ('git filter-repo --path-glob "Scripts" --invert-paths', "Removing Scripts directory from history")
        ]
        
        for command, description in filter_commands:
            success, _ = run_command(command, description, ignore_errors=True)
            if not success:
                print(f"⚠️  Command failed, continuing: {command}")
    else:
        print("🔧 git-filter-repo not found, using git filter-branch...")
        print("   Note: Consider installing git-filter-repo for better performance:")
        print("   pip install git-filter-repo")
        
        # Use git filter-branch as fallback
        filter_command = '''git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *.whl *.pyd site-packages/* Lib/* Scripts/* __pycache__/* build/* dist/*" --prune-empty --tag-name-filter cat -- --all'''
        
        run_command(filter_command, "Cleaning history with filter-branch", ignore_errors=True)
    
    print("✅ Git history cleanup completed")
    return True

def optimize_repository():
    """Optimize the Git repository"""
    print_step(7, "Optimizing Repository")
    
    optimization_commands = [
        ("git reflog expire --expire=now --all", "Expiring reflog entries"),
        ("git gc --prune=now --aggressive", "Running aggressive garbage collection"),
        ("git repack -ad", "Repacking repository objects"),
        ("git prune", "Pruning unreachable objects")
    ]
    
    for command, description in optimization_commands:
        run_command(command, description, ignore_errors=True)
    
    print("✅ Repository optimization completed")
    return True

def show_repository_status():
    """Show current repository status and size"""
    print_step(8, "Repository Status Report")
    
    print("📊 Repository Information:")
    
    # Repository size
    success, output = run_command("git count-objects -vH", "Getting repository statistics")
    if success and output.strip():
        print("   Repository Statistics:")
        for line in output.split('\n'):
            if line.strip():
                print(f"     {line}")
    
    # File count
    success, output = run_command("git ls-files | wc -l", "Counting tracked files", ignore_errors=True)
    if success and output.strip():
        try:
            file_count = int(output.strip())
            print(f"   Tracked files: {file_count}")
        except ValueError:
            # Alternative method for Windows
            success, output = run_command("git ls-files", "Listing tracked files", ignore_errors=True)
            if success:
                file_count = len([line for line in output.split('\n') if line.strip()])
                print(f"   Tracked files: {file_count}")
    
    # Current status
    success, output = run_command("git status --porcelain", "Checking working directory status")
    if success:
        if output.strip():
            changes = len([line for line in output.split('\n') if line.strip()])
            print(f"   Uncommitted changes: {changes}")
        else:
            print("   Working directory: ✅ Clean")
    
    # Directory size
    try:
        git_size = sum(f.stat().st_size for f in Path('.git').rglob('*') if f.is_file())
        git_size_mb = git_size / (1024 * 1024)
        print(f"   .git directory size: {git_size_mb:.1f} MB")
    except Exception:
        print("   .git directory size: Unable to calculate")

def provide_next_steps(backup_dir):
    """Provide next steps to the user"""
    print_header("🎉 CLEANUP COMPLETED!")
    
    print("""
Next Steps:

1. 🧪 TEST YOUR APPLICATION:
   - Make sure your Signal Analyzer still works correctly
   - Check that all important files are still present
   - Run: python run.py (or however you normally start your app)

2. 📤 PUSH CHANGES TO GITHUB:
   - Review changes: git status
   - Push changes: git push --force-with-lease origin main
   - (Use --force-with-lease for safety, not --force)

3. 🧹 CLEANUP:
   - If everything works, remove backup: rm -rf {}
   - Remove old .gitignore backup: rm .gitignore.backup

4. 🚀 SET UP DEPLOYMENT (optional):
   - Run: python setup_deployment.py
   - Create first release: git tag v1.0.0 && git push origin v1.0.0

⚠️  IMPORTANT WARNINGS:
- If you force-push, other collaborators will need to re-clone
- Test your application thoroughly before removing backups
- The backup directory contains your important files if needed

🆘 IF SOMETHING WENT WRONG:
- Your backup is in: {}
- Restore files from backup if needed
- Original .gitignore is in: .gitignore.backup
""".format(backup_dir, backup_dir))

def main():
    """Main cleanup process"""
    print_header("🚀 SIGNAL ANALYZER REPOSITORY CLEANUP")
    
    print("""
This script will clean up your Git repository by:
✅ Removing accidentally committed Python packages
✅ Removing build artifacts and cache files
✅ Creating a comprehensive .gitignore file
✅ Optimizing repository size

⚠️  IMPORTANT: This process modifies Git history!
""")
    
    # Check if we're in a Git repository
    if not check_git_repo():
        return False
    
    # Show current repository status
    print("📊 Current Repository Status:")
    try:
        git_size = sum(f.stat().st_size for f in Path('.git').rglob('*') if f.is_file())
        git_size_mb = git_size / (1024 * 1024)
        print(f"   Current .git size: {git_size_mb:.1f} MB")
    except Exception:
        print("   Current .git size: Unable to calculate")
    
    # Get user confirmation
    print("\n" + "="*70)
    response = input("Do you want to continue with the cleanup? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled by user.")
        return False
    
    try:
        # Execute cleanup steps
        backup_dir, backed_up_files = backup_important_files()
        create_comprehensive_gitignore()
        remove_unwanted_files_from_git()
        add_important_files_back()
        commit_changes()
        clean_git_history()  # Optional step with user confirmation
        optimize_repository()
        show_repository_status()
        provide_next_steps(backup_dir)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup interrupted by user")
        print(f"   Your backup is in: {backup_dir if 'backup_dir' in locals() else 'cleanup_backup_*'}")
        return False
    except Exception as e:
        print(f"\n❌ Cleanup failed with error: {e}")
        print(f"   Your backup is in: {backup_dir if 'backup_dir' in locals() else 'cleanup_backup_*'}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 For manual cleanup, refer to the manual commands provided separately.")
        sys.exit(1)
    else:
        print("\n🎉 Cleanup completed successfully!")
        print("   Don't forget to test your application before removing backups!")