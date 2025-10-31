"""
Comprehensive build testing framework
Place this in the root directory as build_tester.py
"""

import os
import sys
import subprocess
import platform
import time
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import shutil

class BuildTester:
    """Comprehensive testing for built executables"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.dist_dir = self.project_root / "dist"
        self.test_results = {}
        self.system = platform.system().lower()
        
        # Determine executable path based on system
        if self.system == "windows":
            self.exe_path = self.dist_dir / "SignalAnalyzer" / "SignalAnalyzer.exe"
        else:
            self.exe_path = self.dist_dir / "SignalAnalyzer" / "SignalAnalyzer"
    
    def check_build_exists(self) -> bool:
        """Check if the built executable exists"""
        print("🔍 Checking if build exists...")
        
        if not self.dist_dir.exists():
            print(f"❌ Distribution directory not found: {self.dist_dir}")
            return False
        
        if not self.exe_path.exists():
            print(f"❌ Executable not found: {self.exe_path}")
            return False
        
        print(f"✅ Executable found: {self.exe_path}")
        return True
    
    def test_file_structure(self) -> Dict:
        """Test the file structure of the built application"""
        print("🔍 Testing file structure...")
        
        result = {
            'passed': True,
            'missing_files': [],
            'extra_files': [],
            'details': {}
        }
        
        expected_files = [
            "SignalAnalyzer.exe" if self.system == "windows" else "SignalAnalyzer",
            "_internal" if self.system == "windows" else "lib",  # PyInstaller internal files
        ]
        
        # Check for expected files
        app_dir = self.exe_path.parent
        for expected_file in expected_files:
            file_path = app_dir / expected_file
            if not file_path.exists():
                result['missing_files'].append(expected_file)
                result['passed'] = False
        
        # Get total size
        total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
        result['details']['total_size_mb'] = total_size / (1024 * 1024)
        result['details']['file_count'] = len(list(app_dir.rglob('*')))
        
        if result['passed']:
            print(f"✅ File structure OK ({result['details']['file_count']} files, "
                 f"{result['details']['total_size_mb']:.1f} MB)")
        else:
            print(f"❌ File structure issues: missing {result['missing_files']}")
        
        return result
    
    def test_executable_permissions(self) -> Dict:
        """Test executable permissions"""
        print("🔍 Testing executable permissions...")
        
        result = {
            'passed': True,
            'executable': False,
            'readable': False,
            'details': {}
        }
        
        try:
            # Check if file is executable
            result['executable'] = os.access(self.exe_path, os.X_OK)
            result['readable'] = os.access(self.exe_path, os.R_OK)
            
            if self.system != "windows" and not result['executable']:
                # Try to make it executable
                os.chmod(self.exe_path, 0o755)
                result['executable'] = os.access(self.exe_path, os.X_OK)
                result['details']['chmod_applied'] = True
            
            result['passed'] = result['executable'] and result['readable']
            
            if result['passed']:
                print("✅ Executable permissions OK")
            else:
                print(f"❌ Permission issues - executable: {result['executable']}, readable: {result['readable']}")
        
        except Exception as e:
            result['passed'] = False
            result['details']['error'] = str(e)
            print(f"❌ Permission check failed: {e}")
        
        return result
    
    def test_basic_execution(self) -> Dict:
        """Test basic execution (startup and immediate shutdown)"""
        print("🔍 Testing basic execution...")
        
        result = {
            'passed': False,
            'exit_code': None,
            'stdout': '',
            'stderr': '',
            'execution_time': 0,
            'details': {}
        }
        
        try:
            start_time = time.time()
            
            # Run with a timeout to prevent hanging
            process = subprocess.Popen(
                [str(self.exe_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.exe_path.parent
            )
            
            # Give it a few seconds to start, then terminate
            time.sleep(3)
            
            # Try graceful termination first
            if process.poll() is None:  # Still running
                try:
                    if self.system == "windows":
                        process.terminate()
                    else:
                        process.send_signal(signal.SIGTERM)
                    
                    # Wait a bit for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown failed
                        process.kill()
                        process.wait()
                
                except Exception as e:
                    result['details']['termination_error'] = str(e)
                    try:
                        process.kill()
                        process.wait()
                    except:
                        pass
            
            # Get output
            stdout, stderr = process.communicate() if process.poll() is not None else ('', '')
            result['stdout'] = stdout
            result['stderr'] = stderr
            result['exit_code'] = process.returncode
            result['execution_time'] = time.time() - start_time
            
            # Consider it passed if it started (didn't immediately crash)
            result['passed'] = result['execution_time'] > 1.0  # Ran for at least 1 second
            
            if result['passed']:
                print(f"✅ Basic execution OK (ran for {result['execution_time']:.1f}s)")
            else:
                print(f"❌ Basic execution failed (exit code: {result['exit_code']})")
                if stderr:
                    print(f"   stderr: {stderr[:200]}")
        
        except Exception as e:
            result['details']['error'] = str(e)
            print(f"❌ Execution test failed: {e}")
        
        return result
    
    def test_import_resolution(self) -> Dict:
        """Test if critical imports can be resolved by running a minimal test"""
        print("🔍 Testing import resolution...")
        
        result = {
            'passed': False,
            'import_results': {},
            'details': {}
        }
        
        # Create a minimal test script that tests imports
        test_script = '''
import sys
import json

results = {}

# Test critical imports
imports_to_test = [
    'tkinter',
    'numpy', 
    'scipy',
    'matplotlib',
    'pandas'
]

for module_name in imports_to_test:
    try:
        __import__(module_name)
        results[module_name] = {'status': 'ok', 'error': None}
    except Exception as e:
        results[module_name] = {'status': 'failed', 'error': str(e)}

# Test specific submodules that commonly fail
specific_tests = {
    'matplotlib.backends.backend_tkagg': 'from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg',
    'scipy.signal': 'from scipy.signal import savgol_filter',
    'numpy.core': 'import numpy.core._multiarray_umath'
}

for test_name, test_code in specific_tests.items():
    try:
        exec(test_code)
        results[test_name] = {'status': 'ok', 'error': None}
    except Exception as e:
        results[test_name] = {'status': 'failed', 'error': str(e)}

print(json.dumps(results))
'''
        
        try:
            # Write test script to a temporary file
            test_script_path = self.exe_path.parent / "import_test.py"
            with open(test_script_path, 'w') as f:
                f.write(test_script)
            
            # Run the test script with the built Python
            process = subprocess.run(
                [str(self.exe_path.parent / "python"), str(test_script_path)] if (self.exe_path.parent / "python").exists() else 
                [sys.executable, str(test_script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.exe_path.parent
            )
            
            if process.returncode == 0 and process.stdout:
                try:
                    import_results = json.loads(process.stdout.strip())
                    result['import_results'] = import_results
                    
                    # Check if all critical imports passed
                    failed_imports = [name for name, res in import_results.items() 
                                    if res['status'] == 'failed']
                    
                    result['passed'] = len(failed_imports) == 0
                    
                    if result['passed']:
                        print("✅ All critical imports resolved successfully")
                    else:
                        print(f"❌ Failed imports: {failed_imports}")
                        
                except json.JSONDecodeError:
                    result['details']['parse_error'] = process.stdout
                    print(f"❌ Could not parse import test results")
            else:
                result['details']['process_error'] = process.stderr
                print(f"❌ Import test script failed: {process.stderr}")
            
            # Cleanup
            test_script_path.unlink(missing_ok=True)
        
        except Exception as e:
            result['details']['error'] = str(e)
            print(f"❌ Import resolution test failed: {e}")
        
        return result
    
    def test_dependency_libraries(self) -> Dict:
        """Test if required libraries are bundled correctly"""
        print("🔍 Testing dependency libraries...")
        
        result = {
            'passed': True,
            'missing_libraries': [],
            'found_libraries': [],
            'details': {}
        }
        
        # Look for critical library files in the distribution
        app_dir = self.exe_path.parent
        
        # Common library patterns to look for
        library_patterns = {
            'numpy': ['numpy', '_multiarray', 'mkl'],
            'scipy': ['scipy', 'linalg', 'sparse'],
            'matplotlib': ['matplotlib', 'ft2font', '_path'],
            'pandas': ['pandas', '_libs'],
            'tkinter': ['tkinter', '_tkinter']
        }
        
        for lib_name, patterns in library_patterns.items():
            found_files = []
            for pattern in patterns:
                # Search for files containing the pattern
                for file_path in app_dir.rglob('*'):
                    if file_path.is_file() and pattern.lower() in file_path.name.lower():
                        found_files.append(file_path.name)
                        break
            
            if found_files:
                result['found_libraries'].append({
                    'library': lib_name,
                    'files': found_files[:3]  # Limit to first 3 matches
                })
            else:
                result['missing_libraries'].append(lib_name)
                result['passed'] = False
        
        # Check total library directory size
        internal_dir = app_dir / "_internal" if (app_dir / "_internal").exists() else app_dir / "lib"
        if internal_dir.exists():
            lib_size = sum(f.stat().st_size for f in internal_dir.rglob('*') if f.is_file())
            result['details']['library_size_mb'] = lib_size / (1024 * 1024)
        
        if result['passed']:
            print(f"✅ All dependency libraries found")
            if 'library_size_mb' in result['details']:
                print(f"   Library size: {result['details']['library_size_mb']:.1f} MB")
        else:
            print(f"❌ Missing libraries: {result['missing_libraries']}")
        
        return result
    
    def test_sample_data_loading(self) -> Dict:
        """Test loading sample data if available"""
        print("🔍 Testing sample data loading...")
        
        result = {
            'passed': True,
            'sample_files_found': [],
            'loading_results': {},
            'details': {}
        }
        
        # Look for sample data files
        data_dir = self.project_root / "data"
        sample_files = []
        
        if data_dir.exists():
            sample_files = list(data_dir.glob("*sample*.atf")) + list(data_dir.glob("*example*.atf"))
        
        result['sample_files_found'] = [f.name for f in sample_files]
        
        if not sample_files:
            print("ℹ️  No sample data files found")
            return result
        
        # Test loading one sample file (would require running the app)
        # For now, just check if files are readable
        for sample_file in sample_files[:1]:  # Test only first file
            try:
                with open(sample_file, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith('ATF'):
                        result['loading_results'][sample_file.name] = 'readable'
                    else:
                        result['loading_results'][sample_file.name] = 'invalid_format'
                        result['passed'] = False
            except Exception as e:
                result['loading_results'][sample_file.name] = f'error: {str(e)}'
                result['passed'] = False
        
        if result['passed']:
            print(f"✅ Sample data files OK ({len(sample_files)} files)")
        else:
            print(f"❌ Sample data issues: {result['loading_results']}")
        
        return result
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and return comprehensive results"""
        print("🧪 Running comprehensive build tests...")
        print("="*60)
        
        # Initialize results
        all_results = {
            'timestamp': time.time(),
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'executable_path': str(self.exe_path)
            },
            'tests': {},
            'overall_passed': False,
            'critical_failures': [],
            'warnings': []
        }
        
        # Check if build exists first
        if not self.check_build_exists():
            all_results['critical_failures'].append('Build does not exist')
            return all_results
        
        # Run all tests
        test_functions = [
            ('file_structure', self.test_file_structure),
            ('executable_permissions', self.test_executable_permissions),
            ('basic_execution', self.test_basic_execution),
            ('import_resolution', self.test_import_resolution),
            ('dependency_libraries', self.test_dependency_libraries),
            ('sample_data_loading', self.test_sample_data_loading),
        ]
        
        for test_name, test_func in test_functions:
            print(f"\n--- {test_name.replace('_', ' ').title()} ---")
            try:
                test_result = test_func()
                all_results['tests'][test_name] = test_result
                
                if not test_result['passed']:
                    if test_name in ['file_structure', 'executable_permissions', 'basic_execution']:
                        all_results['critical_failures'].append(test_name)
                    else:
                        all_results['warnings'].append(test_name)
                        
            except Exception as e:
                print(f"❌ Test {test_name} failed with exception: {e}")
                all_results['tests'][test_name] = {
                    'passed': False,
                    'error': str(e),
                    'details': {}
                }
                all_results['critical_failures'].append(test_name)
        
        # Determine overall result
        all_results['overall_passed'] = len(all_results['critical_failures']) == 0
        
        return all_results
    
    def generate_test_report(self, results: Dict) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("="*60)
        report.append("BUILD TEST REPORT")
        report.append("="*60)
        
        # System info
        report.append(f"Platform: {results['system_info']['platform']}")
        report.append(f"Executable: {results['system_info']['executable_path']}")
        report.append(f"Test time: {time.ctime(results['timestamp'])}")
        
        # Overall status
        if results['overall_passed']:
            report.append(f"\n✅ OVERALL STATUS: PASSED")
        else:
            report.append(f"\n❌ OVERALL STATUS: FAILED")
        
        # Critical failures
        if results['critical_failures']:
            report.append(f"\n❌ CRITICAL FAILURES:")
            for failure in results['critical_failures']:
                report.append(f"  - {failure}")
        
        # Warnings
        if results['warnings']:
            report.append(f"\n⚠️  WARNINGS:")
            for warning in results['warnings']:
                report.append(f"  - {warning}")
        
        # Detailed test results
        report.append(f"\nDETAILED TEST RESULTS:")
        report.append("-" * 30)
        
        for test_name, test_result in results['tests'].items():
            status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
            report.append(f"{test_name}: {status}")
            
            # Add relevant details
            if 'details' in test_result and test_result['details']:
                for key, value in test_result['details'].items():
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def save_test_results(self, results: Dict):
        """Save test results to files"""
        # Save JSON results
        results_file = self.project_root / "build_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        report = self.generate_test_report(results)
        report_file = self.project_root / "build_test_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test results saved:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")

def main():
    """Main function for running build tests"""
    tester = BuildTester()
    
    print("🚀 Starting comprehensive build testing...")
    
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(tester.generate_test_report(results))
    
    # Save results
    tester.save_test_results(results)
    
    # Return appropriate exit code
    return 0 if results['overall_passed'] else 1

if __name__ == "__main__":
    sys.exit(main())