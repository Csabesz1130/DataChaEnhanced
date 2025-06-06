import os
import sys

# Add the 'src' directory to the Python path to allow imports
# This is necessary because the script is in the root directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def check_file(filepath):
    """
    Checks a single file for null bytes and attempts to decode it as UTF-8.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return

    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                print(f"❌ ERROR: Found null bytes in '{filepath}'!")
            else:
                print(f"✅ OK: No null bytes found in '{filepath}'.")
                
                # Check if it's valid UTF-8
                try:
                    content.decode('utf-8')
                    print(f"   └── ✅ OK: File is valid UTF-8.")
                except UnicodeDecodeError:
                    print(f"   └── ⚠️ WARNING: File is NOT valid UTF-8. Please re-save it.")

    except Exception as e:
        print(f"❌ ERROR: Could not read file '{filepath}': {e}")

def main():
    print("--- Running file encoding and import checks ---")
    
    # List of files in the import chain for the AI Analysis Tab
    files_to_check = [
        os.path.join('src', 'gui', 'app.py'),
        os.path.join('src', 'gui', 'ai_analysis_tab.py'),
        os.path.join('src', 'analysis', 'ai_integral_calculator.py'),
        os.path.join('src', 'config', 'ai_config.py'),
        os.path.join('src', 'utils', 'logger.py')
    ]
    
    print("\n--- 1. Checking files for null bytes and UTF-8 encoding ---")
    for f in files_to_check:
        check_file(f)
        
    print("\n--- 2. Attempting to import the main problematic module ---")
    try:
        print("Importing 'src.gui.ai_analysis_tab'...")
        # We import it inside a function to avoid it being cached
        import src.gui.ai_analysis_tab
        print("\n✅ SUCCESS: 'ai_analysis_tab' imported correctly.")
        print("The issue might be with how it's integrated in app.py.")
    except Exception as e:
        print(f"\n❌ FAILED to import 'ai_analysis_tab'.")
        print(f"   └── Error Type: {type(e).__name__}")
        print(f"   └── Error Message: {e}")

if __name__ == "__main__":
    main()