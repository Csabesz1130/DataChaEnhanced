#!/usr/bin/env python3
"""
Advanced Repository Size Analyzer - Windows/PowerShell Compatible

This script analyzes your repository to identify what's causing size issues,
provides detailed insights about large files/folders, and gives recommendations
for cleanup.

Compatible with Windows, PowerShell, and Unix systems.
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path
from collections import defaultdict, namedtuple
import argparse
from datetime import datetime

# ANSI color codes for beautiful output (works in modern PowerShell)
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Emoji-like symbols for better visualization
    WARNING = '⚠️ '
    ERROR = '❌'
    SUCCESS = '✅'
    INFO = 'ℹ️ '
    FOLDER = '📁'
    FILE = '📄'
    GIT = '🔄'
    SIZE = '📊'
    CLEAN = '🧹'

# Check if we're on Windows and if ANSI colors are supported
if platform.system() == 'Windows':
    try:
        # Enable ANSI color support on Windows 10+
        import colorama
        colorama.init()
    except ImportError:
        # If colorama not available, disable colors
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')

FileInfo = namedtuple('FileInfo', ['path', 'size', 'type', 'git_status', 'should_ignore'])

class RepoSizeAnalyzer:
    def __init__(self, repo_path='.'):
        self.repo_path = Path(repo_path).resolve()
        self.total_size = 0
        self.file_categories = defaultdict(list)
        self.large_files = []
        self.large_dirs = []
        self.git_objects = []
        self.recommendations = []
        self.is_windows = platform.system() == 'Windows'
        
        # Size thresholds (in bytes)
        self.LARGE_FILE_THRESHOLD = 1024 * 1024  # 1MB
        self.LARGE_DIR_THRESHOLD = 10 * 1024 * 1024  # 10MB
        self.HUGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50MB
        
        # Patterns for different file types
        self.IGNORE_PATTERNS = {
            'dependencies': ['node_modules', 'bower_components', 'vendor', '.venv', 'venv', 'env', 'packages'],
            'cache': ['__pycache__', '.pytest_cache', '.cache', '.npm', '.yarn', 'node_modules/.cache'],
            'build': ['build', 'dist', 'target', 'out', '*.egg-info', 'bin', 'obj'],
            'ide': ['.vscode', '.idea', '.vs', '*.sublime-*', '.vscode-test'],
            'os': ['.DS_Store', 'Thumbs.db', 'desktop.ini', '$RECYCLE.BIN'],
            'logs': ['*.log', 'logs', 'log', '*.log.*'],
            'temp': ['tmp', 'temp', '*.tmp', '*.temp', '*.bak'],
            'compiled': ['*.pyc', '*.pyo', '*.pyd', '*.class', '*.o', '*.so', '*.dll', '*.exe', '*.msi']
        }

    def format_size(self, size_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def run_command(self, command, shell=True, check=False):
        """Run a system command and return the output."""
        try:
            # Adjust command for Windows if needed
            if self.is_windows and command.startswith('git'):
                # Make sure git commands work on Windows
                command = command.replace("'", '"')
            
            result = subprocess.run(
                command, 
                shell=shell, 
                capture_output=True, 
                text=True, 
                cwd=self.repo_path,
                check=check,
                timeout=30
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    def run_git_command(self, command, check=False):
        """Run a git command and return the output."""
        return self.run_command(f"git {command}", check=check)

    def is_git_repo(self):
        """Check if the current directory is a git repository."""
        return (self.repo_path / '.git').exists() or self.run_git_command('rev-parse --git-dir') is not None

    def get_git_status(self, file_path):
        """Get git status of a file."""
        if not self.is_git_repo():
            return 'not_git_repo'
        
        try:
            relative_path = file_path.relative_to(self.repo_path)
            # Use forward slashes for git even on Windows
            git_path = str(relative_path).replace('\\', '/')
            status = self.run_git_command(f'status --porcelain "{git_path}"')
            
            if status is None:
                return 'error'
            elif status == '':
                return 'tracked'
            elif status.startswith('??'):
                return 'untracked'
            elif status.startswith('A'):
                return 'added'
            elif status.startswith('M'):
                return 'modified'
            elif status.startswith('D'):
                return 'deleted'
            else:
                return 'other'
        except Exception:
            return 'error'

    def should_be_ignored(self, path):
        """Check if a file/directory should be ignored."""
        path_str = str(path).lower().replace('\\', '/')
        path_name = path.name.lower()
        
        for category, patterns in self.IGNORE_PATTERNS.items():
            for pattern in patterns:
                if pattern.startswith('*'):
                    if path_name.endswith(pattern[1:]):
                        return category
                elif pattern in path_str or pattern == path_name:
                    return category
        return None

    def get_directory_size(self, path):
        """Calculate total size of a directory - Windows compatible."""
        total = 0
        try:
            # Try fast method first (Windows PowerShell or Unix)
            if self.is_windows:
                # PowerShell command for Windows
                cmd = f'powershell -Command "(Get-ChildItem -Path \'{path}\' -Recurse -File | Measure-Object -Property Length -Sum).Sum"'
                result = self.run_command(cmd)
                if result and result.strip().isdigit():
                    return int(result.strip())
            else:
                # Unix du command
                result = self.run_command(f'du -sb "{path}"')
                if result:
                    return int(result.split()[0])
        except:
            pass
        
        # Fallback to Python calculation
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = Path(dirpath) / filename
                    try:
                        total += file_path.stat().st_size
                    except (OSError, FileNotFoundError, PermissionError):
                        continue
        except (OSError, PermissionError):
            pass
        return total

    def analyze_directory_structure(self):
        """Analyze the directory structure and categorize files."""
        print(f"{Colors.INFO}Analyzing directory structure...")
        
        for root, dirs, files in os.walk(self.repo_path):
            root_path = Path(root)
            
            # Skip .git directory for file analysis (but we'll analyze it separately)
            if '.git' in root_path.parts:
                continue
                
            # Analyze directories
            for dir_name in dirs:
                dir_path = root_path / dir_name
                if dir_path.name == '.git':
                    continue
                    
                try:
                    dir_size = self.get_directory_size(dir_path)
                    ignore_reason = self.should_be_ignored(dir_path)
                    git_status = self.get_git_status(dir_path)
                    
                    file_info = FileInfo(
                        path=dir_path,
                        size=dir_size,
                        type='directory',
                        git_status=git_status,
                        should_ignore=ignore_reason
                    )
                    
                    if ignore_reason:
                        self.file_categories[ignore_reason].append(file_info)
                    
                    if dir_size > self.LARGE_DIR_THRESHOLD:
                        self.large_dirs.append(file_info)
                        
                except (OSError, PermissionError):
                    continue
            
            # Analyze files
            for file_name in files:
                file_path = root_path / file_name
                try:
                    file_size = file_path.stat().st_size
                    ignore_reason = self.should_be_ignored(file_path)
                    git_status = self.get_git_status(file_path)
                    
                    file_info = FileInfo(
                        path=file_path,
                        size=file_size,
                        type='file',
                        git_status=git_status,
                        should_ignore=ignore_reason
                    )
                    
                    if ignore_reason:
                        self.file_categories[ignore_reason].append(file_info)
                    
                    if file_size > self.LARGE_FILE_THRESHOLD:
                        self.large_files.append(file_info)
                        
                except (OSError, PermissionError):
                    continue

    def analyze_git_objects(self):
        """Analyze Git objects to find large files in history."""
        if not self.is_git_repo():
            return
            
        print(f"{Colors.INFO}Analyzing Git objects...")
        
        # Get all objects sorted by size
        objects_output = self.run_git_command('rev-list --objects --all')
        
        if objects_output:
            # Process objects in batches to avoid command line length limits
            lines = objects_output.split('\n')[:500]  # Limit to first 500 objects for performance
            
            for line in lines:
                if line.strip():
                    parts = line.split(' ', 1)
                    if len(parts) >= 1:
                        obj_hash = parts[0]
                        filename = parts[1] if len(parts) > 1 else 'unknown'
                        
                        # Get object size
                        size_output = self.run_git_command(f'cat-file -s {obj_hash}')
                        if size_output and size_output.isdigit():
                            size = int(size_output)
                            
                            if size > self.LARGE_FILE_THRESHOLD:
                                self.git_objects.append({
                                    'filename': filename,
                                    'size': size,
                                    'hash': obj_hash
                                })
        
        # Sort by size (largest first)
        self.git_objects.sort(key=lambda x: x['size'], reverse=True)

    def get_repo_stats(self):
        """Get overall repository statistics."""
        stats = {}
        
        # Working directory size
        stats['working_dir_size'] = self.get_directory_size(self.repo_path)
        
        # Git directory size
        git_dir = self.repo_path / '.git'
        if git_dir.exists():
            stats['git_dir_size'] = self.get_directory_size(git_dir)
        else:
            stats['git_dir_size'] = 0
            
        stats['total_size'] = stats['working_dir_size'] + stats['git_dir_size']
        
        # File counts
        stats['total_files'] = sum(len(files) for _, _, files in os.walk(self.repo_path))
        stats['total_dirs'] = sum(len(dirs) for _, dirs, _ in os.walk(self.repo_path))
        
        return stats

    def generate_recommendations(self):
        """Generate cleanup recommendations based on analysis."""
        self.recommendations = []
        
        # Check for large ignored directories
        for category, files in self.file_categories.items():
            if not files:
                continue
                
            total_size = sum(f.size for f in files)
            file_count = len(files)
            
            if total_size > 10 * 1024 * 1024:  # > 10MB
                if category == 'dependencies':
                    cmd = 'Remove-Item -Recurse -Force node_modules, bower_components, vendor' if self.is_windows else 'rm -rf node_modules/ bower_components/ vendor/'
                    self.recommendations.append({
                        'priority': 'HIGH',
                        'action': f'Remove {category} directories',
                        'description': f'Found {file_count} dependency directories using {self.format_size(total_size)}',
                        'command': cmd,
                        'savings': total_size
                    })
                elif category == 'cache':
                    if self.is_windows:
                        cmd = 'Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force'
                    else:
                        cmd = 'find . -name "__pycache__" -type d -exec rm -rf {} +'
                    self.recommendations.append({
                        'priority': 'HIGH',
                        'action': f'Remove {category} files',
                        'description': f'Found {file_count} cache files/dirs using {self.format_size(total_size)}',
                        'command': cmd,
                        'savings': total_size
                    })
                elif category == 'build':
                    cmd = 'Remove-Item -Recurse -Force build, dist, target' if self.is_windows else 'rm -rf build/ dist/ target/ *.egg-info/'
                    self.recommendations.append({
                        'priority': 'MEDIUM',
                        'action': f'Remove {category} artifacts',
                        'description': f'Found {file_count} build artifacts using {self.format_size(total_size)}',
                        'command': cmd,
                        'savings': total_size
                    })
        
        # Check for large tracked files that shouldn't be tracked
        large_tracked = [f for f in self.large_files if f.git_status == 'tracked' and f.should_ignore]
        if large_tracked:
            total_size = sum(f.size for f in large_tracked)
            self.recommendations.append({
                'priority': 'HIGH',
                'action': 'Remove large tracked files that should be ignored',
                'description': f'Found {len(large_tracked)} large files that are tracked but should be ignored',
                'command': 'git rm --cached <filename>',
                'savings': total_size
            })
        
        # Check for huge files in git history
        if self.git_objects:
            huge_objects = [obj for obj in self.git_objects[:5] if obj['size'] > self.HUGE_FILE_THRESHOLD]
            if huge_objects:
                total_size = sum(obj['size'] for obj in huge_objects)
                self.recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Consider using git-filter-branch or BFG',
                    'description': f'Found {len(huge_objects)} huge files in git history',
                    'command': 'git filter-branch or BFG Repo-Cleaner',
                    'savings': total_size
                })

    def print_header(self):
        """Print a beautiful header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
        print(f"{Colors.BOLD}{Colors.CYAN}🔍 REPOSITORY SIZE ANALYZER 🔍")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
        print(f"{Colors.INFO}Repository: {Colors.BOLD}{self.repo_path}{Colors.END}")
        print(f"{Colors.INFO}Platform: {Colors.BOLD}{platform.system()} {platform.release()}{Colors.END}")
        print(f"{Colors.INFO}Analysis Time: {Colors.BOLD}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")

    def print_summary(self, stats):
        """Print repository summary."""
        print(f"{Colors.BOLD}{Colors.BLUE}📊 REPOSITORY SUMMARY{Colors.END}")
        print(f"{Colors.BLUE}{'─'*50}{Colors.END}")
        
        print(f"{Colors.SIZE} Total Repository Size: {Colors.BOLD}{Colors.RED}{self.format_size(stats['total_size'])}{Colors.END}")
        print(f"  ├─ Working Directory: {Colors.YELLOW}{self.format_size(stats['working_dir_size'])}{Colors.END}")
        print(f"  └─ Git Directory (.git): {Colors.YELLOW}{self.format_size(stats['git_dir_size'])}{Colors.END}")
        
        print(f"\n{Colors.FILE} Files & Directories:")
        print(f"  ├─ Total Files: {Colors.GREEN}{stats['total_files']:,}{Colors.END}")
        print(f"  └─ Total Directories: {Colors.GREEN}{stats['total_dirs']:,}{Colors.END}")

    def print_large_directories(self):
        """Print information about large directories."""
        if not self.large_dirs:
            return
            
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}📁 LARGE DIRECTORIES (>{self.format_size(self.LARGE_DIR_THRESHOLD)}){Colors.END}")
        print(f"{Colors.MAGENTA}{'─'*70}{Colors.END}")
        
        # Sort by size (largest first)
        sorted_dirs = sorted(self.large_dirs, key=lambda x: x.size, reverse=True)
        
        for i, dir_info in enumerate(sorted_dirs[:10]):  # Show top 10
            relative_path = dir_info.path.relative_to(self.repo_path)
            size_color = Colors.RED if dir_info.size > 100*1024*1024 else Colors.YELLOW
            
            status_icon = {
                'tracked': '🔄',
                'untracked': '❓',
                'not_git_repo': '📁'
            }.get(dir_info.git_status, '❓')
            
            ignore_info = f" ({Colors.RED}SHOULD BE IGNORED: {dir_info.should_ignore}{Colors.END})" if dir_info.should_ignore else ""
            
            print(f"{i+1:2d}. {status_icon} {size_color}{self.format_size(dir_info.size):<10}{Colors.END} {relative_path}{ignore_info}")

    def print_large_files(self):
        """Print information about large files."""
        if not self.large_files:
            return
            
        print(f"\n{Colors.BOLD}{Colors.YELLOW}📄 LARGE FILES (>{self.format_size(self.LARGE_FILE_THRESHOLD)}){Colors.END}")
        print(f"{Colors.YELLOW}{'─'*70}{Colors.END}")
        
        # Sort by size (largest first)
        sorted_files = sorted(self.large_files, key=lambda x: x.size, reverse=True)
        
        for i, file_info in enumerate(sorted_files[:15]):  # Show top 15
            relative_path = file_info.path.relative_to(self.repo_path)
            size_color = Colors.RED if file_info.size > self.HUGE_FILE_THRESHOLD else Colors.YELLOW
            
            status_icon = {
                'tracked': '🔄',
                'untracked': '❓',
                'modified': '✏️',
                'not_git_repo': '📄'
            }.get(file_info.git_status, '❓')
            
            ignore_info = f" ({Colors.RED}SHOULD BE IGNORED: {file_info.should_ignore}{Colors.END})" if file_info.should_ignore else ""
            
            print(f"{i+1:2d}. {status_icon} {size_color}{self.format_size(file_info.size):<10}{Colors.END} {relative_path}{ignore_info}")

    def print_category_analysis(self):
        """Print analysis by file categories."""
        if not any(self.file_categories.values()):
            return
            
        print(f"\n{Colors.BOLD}{Colors.CYAN}🗂️  CATEGORY ANALYSIS{Colors.END}")
        print(f"{Colors.CYAN}{'─'*70}{Colors.END}")
        
        for category, files in self.file_categories.items():
            if not files:
                continue
                
            total_size = sum(f.size for f in files)
            file_count = len(files)
            dir_count = len([f for f in files if f.type == 'directory'])
            
            severity = Colors.RED if total_size > 50*1024*1024 else Colors.YELLOW if total_size > 10*1024*1024 else Colors.GREEN
            
            print(f"\n{Colors.WARNING}{severity}{category.upper()}{Colors.END}")
            print(f"  📊 Total Size: {severity}{self.format_size(total_size)}{Colors.END}")
            print(f"  📁 Directories: {dir_count}")
            print(f"  📄 Files: {file_count - dir_count}")
            
            # Show largest items in this category
            largest_items = sorted(files, key=lambda x: x.size, reverse=True)[:3]
            for item in largest_items:
                relative_path = item.path.relative_to(self.repo_path)
                print(f"    └─ {self.format_size(item.size):<10} {relative_path}")

    def print_git_objects(self):
        """Print information about large Git objects."""
        if not self.git_objects:
            return
            
        print(f"\n{Colors.BOLD}{Colors.GREEN}🔄 LARGE GIT OBJECTS{Colors.END}")
        print(f"{Colors.GREEN}{'─'*70}{Colors.END}")
        
        total_size = sum(obj['size'] for obj in self.git_objects)
        print(f"Total size of large objects in Git history: {Colors.BOLD}{self.format_size(total_size)}{Colors.END}\n")
        
        for i, obj in enumerate(self.git_objects[:10]):  # Show top 10
            size_color = Colors.RED if obj['size'] > self.HUGE_FILE_THRESHOLD else Colors.YELLOW
            print(f"{i+1:2d}. {size_color}{self.format_size(obj['size']):<10}{Colors.END} {obj['filename']}")

    def print_recommendations(self):
        """Print cleanup recommendations."""
        if not self.recommendations:
            return
            
        print(f"\n{Colors.BOLD}{Colors.GREEN}🧹 CLEANUP RECOMMENDATIONS{Colors.END}")
        print(f"{Colors.GREEN}{'─'*70}{Colors.END}")
        
        # Sort by priority and potential savings
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_recs = sorted(self.recommendations, 
                           key=lambda x: (priority_order.get(x['priority'], 3), -x['savings']))
        
        total_potential_savings = sum(rec['savings'] for rec in sorted_recs)
        print(f"💰 Potential Space Savings: {Colors.BOLD}{Colors.GREEN}{self.format_size(total_potential_savings)}{Colors.END}\n")
        
        shell_type = "PowerShell" if self.is_windows else "Bash"
        print(f"Commands for {shell_type}:\n")
        
        for i, rec in enumerate(sorted_recs):
            priority_color = {
                'HIGH': Colors.RED,
                'MEDIUM': Colors.YELLOW,
                'LOW': Colors.GREEN
            }.get(rec['priority'], Colors.WHITE)
            
            print(f"{i+1}. {priority_color}{rec['priority']} PRIORITY{Colors.END}: {rec['action']}")
            print(f"   📝 {rec['description']}")
            print(f"   💾 Potential Savings: {Colors.GREEN}{self.format_size(rec['savings'])}{Colors.END}")
            print(f"   💻 Command: {Colors.CYAN}{rec['command']}{Colors.END}\n")

    def print_powershell_tips(self):
        """Print PowerShell-specific tips."""
        if not self.is_windows:
            return
            
        print(f"\n{Colors.BOLD}{Colors.BLUE}💡 POWERSHELL TIPS{Colors.END}")
        print(f"{Colors.BLUE}{'─'*50}{Colors.END}")
        print(f"• Run PowerShell as Administrator for better performance")
        print(f"• Use Tab completion for file paths")
        print(f"• Add {Colors.CYAN}-WhatIf{Colors.END} to Remove-Item commands to preview")
        print(f"• Use {Colors.CYAN}Get-ChildItem -Recurse | Measure-Object -Property Length -Sum{Colors.END} to check sizes")

    def export_report(self, filename='repo_analysis.json'):
        """Export detailed analysis to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'repository_path': str(self.repo_path),
            'platform': platform.system(),
            'stats': self.get_repo_stats(),
            'large_directories': [
                {
                    'path': str(d.path.relative_to(self.repo_path)),
                    'size': d.size,
                    'git_status': d.git_status,
                    'should_ignore': d.should_ignore
                } for d in sorted(self.large_dirs, key=lambda x: x.size, reverse=True)
            ],
            'large_files': [
                {
                    'path': str(f.path.relative_to(self.repo_path)),
                    'size': f.size,
                    'git_status': f.git_status,
                    'should_ignore': f.should_ignore
                } for f in sorted(self.large_files, key=lambda x: x.size, reverse=True)
            ],
            'git_objects': self.git_objects,
            'recommendations': self.recommendations,
            'categories': {
                category: {
                    'total_size': sum(f.size for f in files),
                    'count': len(files),
                    'items': [
                        {
                            'path': str(f.path.relative_to(self.repo_path)),
                            'size': f.size,
                            'type': f.type
                        } for f in sorted(files, key=lambda x: x.size, reverse=True)[:5]
                    ]
                } for category, files in self.file_categories.items() if files
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{Colors.SUCCESS} Detailed report exported to: {Colors.BOLD}{filename}{Colors.END}")

    def run_analysis(self, export_json=False):
        """Run the complete analysis."""
        self.print_header()
        
        # Run analysis
        stats = self.get_repo_stats()
        self.analyze_directory_structure()
        if self.is_git_repo():
            self.analyze_git_objects()
        self.generate_recommendations()
        
        # Print results
        self.print_summary(stats)
        self.print_large_directories()
        self.print_large_files()
        self.print_category_analysis()
        if self.is_git_repo():
            self.print_git_objects()
        self.print_recommendations()
        
        # PowerShell-specific tips
        if self.is_windows:
            self.print_powershell_tips()
        
        # Export if requested
        if export_json:
            self.export_report()
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Analysis Complete!{Colors.END}")
        if not export_json:
            print(f"{Colors.INFO}Run with --export to save detailed JSON report{Colors.END}")
        print(f"{Colors.INFO}Platform: {platform.system()}{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Repository Size Analyzer - Windows/PowerShell Compatible",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python repo_size_analyzer.py                    # Analyze current directory
  python repo_size_analyzer.py --path C:\\repos\\myproject  # Windows path
  python repo_size_analyzer.py --export           # Export detailed JSON report
  python repo_size_analyzer.py --threshold 5M     # Set custom file size threshold

PowerShell Examples:
  python repo_size_analyzer.py --path "C:\\Users\\Name\\Documents\\MyRepo"
  python repo_size_analyzer.py --threshold 10MB --export
        """
    )
    
    parser.add_argument('--path', '-p', default='.', 
                       help='Path to repository (default: current directory)')
    parser.add_argument('--export', '-e', action='store_true',
                       help='Export detailed analysis to JSON file')
    parser.add_argument('--threshold', '-t', default='1M',
                       help='Large file threshold (e.g., 1M, 10MB, 0.5G)')
    
    args = parser.parse_args()
    
    # Parse threshold
    threshold_str = args.threshold.upper()
    multiplier = {'K': 1024, 'M': 1024**2, 'MB': 1024**2, 'G': 1024**3, 'GB': 1024**3}.get(threshold_str[-2:], 1)
    if threshold_str[-2:] in ['MB', 'GB']:
        threshold_value = float(threshold_str[:-2]) * multiplier
    elif threshold_str[-1] in 'KMG':
        multiplier = {'K': 1024, 'M': 1024**2, 'G': 1024**3}[threshold_str[-1]]
        threshold_value = float(threshold_str[:-1]) * multiplier
    else:
        threshold_value = float(threshold_str)
    
    # Run analysis
    try:
        analyzer = RepoSizeAnalyzer(args.path)
        analyzer.LARGE_FILE_THRESHOLD = int(threshold_value)
        analyzer.run_analysis(export_json=args.export)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.ERROR} Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()