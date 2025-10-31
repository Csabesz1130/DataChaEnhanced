#!/usr/bin/env python3
"""
Disk Space Checker for DataChaEnhanced

This script checks available disk space and provides guidance for installing
AI/ML dependencies like TensorFlow.
"""

import os
import shutil
import platform

def get_disk_usage(path="."):
    """Get disk usage information for the given path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'free_percent': (free / total) * 100
        }
    except Exception as e:
        return {'error': str(e)}

def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def main():
    print("=" * 60)
    print("DataChaEnhanced - Disk Space Checker")
    print("=" * 60)
    
    # Get current directory disk usage
    current_dir = os.path.abspath(".")
    print(f"Current directory: {current_dir}")
    
    # Get disk usage for current directory
    usage = get_disk_usage(current_dir)
    
    if 'error' in usage:
        print(f"Error checking disk space: {usage['error']}")
        return
    
    print(f"\nDisk Space Information:")
    print(f"  Total: {usage['total_gb']:.2f} GB")
    print(f"  Used:  {usage['used_gb']:.2f} GB")
    print(f"  Free:  {usage['free_gb']:.2f} GB ({usage['free_percent']:.1f}%)")
    
    # Check if we're on Windows and get system drive info
    if platform.system() == "Windows":
        system_drive = os.path.splitdrive(current_dir)[0] + "\\"
        print(f"\nSystem drive: {system_drive}")
        
        system_usage = get_disk_usage(system_drive)
        if 'error' not in system_usage:
            print(f"System drive space:")
            print(f"  Total: {system_usage['total_gb']:.2f} GB")
            print(f"  Used:  {system_usage['used_gb']:.2f} GB")
            print(f"  Free:  {system_usage['free_gb']:.2f} GB ({system_usage['free_percent']:.1f}%)")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if usage['free_gb'] < 5:
        print("  ⚠️  CRITICAL: Less than 5 GB free space!")
        print("     - TensorFlow requires at least 2-3 GB for installation")
        print("     - OpenCV (cv2) requires additional space")
        print("     - Consider freeing up space before installing AI/ML libraries")
    elif usage['free_gb'] < 10:
        print("  ⚠️  WARNING: Less than 10 GB free space")
        print("     - TensorFlow installation may fail")
        print("     - Consider freeing up some space")
    else:
        print("  ✅ Sufficient disk space available")
        print("     - TensorFlow and other AI/ML libraries can be installed")
    
    print(f"\nSpace Requirements:")
    print(f"  - TensorFlow: ~2-3 GB")
    print(f"  - OpenCV (cv2): ~500 MB")
    print(f"  - Other AI/ML libraries: ~1-2 GB")
    print(f"  - Total recommended: ~5-6 GB free space")
    
    print(f"\nHow to free up space:")
    print(f"  1. Delete temporary files (Windows: %TEMP%)")
    print(f"  2. Clear browser cache and downloads")
    print(f"  3. Uninstall unused programs")
    print(f"  4. Move large files to external storage")
    print(f"  5. Use Disk Cleanup utility (Windows)")
    
    print(f"\nAlternative solutions:")
    print(f"  1. Use the simplified background processor (no TensorFlow required)")
    print(f"  2. Install only essential libraries (pandas, numpy, openpyxl)")
    print(f"  3. Use cloud-based AI services instead of local models")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
