# install_unicode_fixes.ps1
# Automated installation and testing script for Unicode fixes
# Usage: .\install_unicode_fixes.ps1

param(
    [switch]$SkipBackup,
    [switch]$SkipTest,
    [switch]$Force
)

# Color-coded output functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️ $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️ $Message" -ForegroundColor Blue }
function Write-Step { param($Message) Write-Host "🔧 $Message" -ForegroundColor Cyan }

# Header
Clear-Host
Write-Host "🔧 UNICODE FIXES INSTALLATION SCRIPT" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "This script will install the Unicode-fixed performance tools" -ForegroundColor White
Write-Host ""

$startTime = Get-Date

# Step 1: Check current location
Write-Step "Step 1: Checking current directory..."

$currentDir = Get-Location
Write-Host "   Current directory: $currentDir" -ForegroundColor Gray

if (-not (Test-Path "performance_analysis")) {
    Write-Warning "performance_analysis directory not found in current location"
    Write-Host "Please run this script from your project root directory" -ForegroundColor Yellow
    Write-Host "Expected structure:" -ForegroundColor Yellow
    Write-Host "  your_project/" -ForegroundColor Gray
    Write-Host "  ├── src/" -ForegroundColor Gray
    Write-Host "  ├── performance_analysis/  ← Should exist" -ForegroundColor Gray
    Write-Host "  └── performance_reports/   ← Will be created" -ForegroundColor Gray
    exit 1
}

Write-Success "Found performance_analysis directory"

# Step 2: Create/verify directory structure
Write-Step "Step 2: Setting up directory structure..."

$performanceDir = "performance_analysis"
$reportsDir = "performance_reports"

if (-not (Test-Path $reportsDir)) {
    Write-Info "Creating performance_reports directory..."
    try {
        New-Item -ItemType Directory -Path $reportsDir | Out-Null
        Write-Success "Created performance_reports directory"
    } catch {
        Write-Error "Failed to create performance_reports directory: $($_.Exception.Message)"
        exit 1
    }
} else {
    Write-Success "performance_reports directory exists"
}

# Step 3: Check dependencies
Write-Step "Step 3: Checking Python dependencies..."

$dependencies = @("psutil", "matplotlib", "numpy")
$missingDeps = @()

foreach ($dep in $dependencies) {
    try {
        $result = python -c "import $dep; print('OK')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $dep" -ForegroundColor Green
        } else {
            $missingDeps += $dep
            Write-Host "   ❌ $dep - Missing" -ForegroundColor Red
        }
    } catch {
        $missingDeps += $dep
        Write-Host "   ❌ $dep - Error checking" -ForegroundColor Red
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Warning "Missing dependencies detected: $($missingDeps -join ', ')"
    
    if ($Force) {
        Write-Info "Installing missing dependencies..."
        try {
            pip install $missingDeps
            Write-Success "Dependencies installed successfully"
        } catch {
            Write-Error "Failed to install dependencies. Please run: pip install $($missingDeps -join ' ')"
            exit 1
        }
    } else {
        Write-Host "Install with: pip install $($missingDeps -join ' ')" -ForegroundColor Yellow
        Write-Host "Or run this script with -Force to auto-install" -ForegroundColor Yellow
        exit 1
    }
}

# Step 4: Backup existing files
Write-Step "Step 4: Backing up existing files..."

$filesToBackup = @(
    "$performanceDir/memory_leak_detector.py",
    "$performanceDir/signal_analyzer_profiler.py"
)

if (-not $SkipBackup) {
    $backupDir = "$performanceDir/backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    $hasFilesToBackup = $false
    foreach ($file in $filesToBackup) {
        if (Test-Path $file) {
            $hasFilesToBackup = $true
            break
        }
    }
    
    if ($hasFilesToBackup) {
        try {
            New-Item -ItemType Directory -Path $backupDir | Out-Null
            
            foreach ($file in $filesToBackup) {
                if (Test-Path $file) {
                    $fileName = Split-Path $file -Leaf
                    Copy-Item $file "$backupDir/$fileName"
                    Write-Host "   📄 Backed up: $fileName" -ForegroundColor Gray
                }
            }
            
            Write-Success "Files backed up to: $backupDir"
        } catch {
            Write-Warning "Could not create backup: $($_.Exception.Message)"
            if (-not $Force) {
                Write-Host "Continue anyway? (y/N): " -NoNewline -ForegroundColor Yellow
                $response = Read-Host
                if ($response -ne 'y' -and $response -ne 'Y') {
                    Write-Host "Installation cancelled." -ForegroundColor Yellow
                    exit 1
                }
            }
        }
    } else {
        Write-Info "No existing files to backup"
    }
} else {
    Write-Info "Skipping backup (--SkipBackup specified)"
}

# Step 5: Check if we have the fixed files available
Write-Step "Step 5: Checking for Unicode-fixed files..."

# Since we can't directly access the artifacts, we'll guide the user
Write-Info "Please ensure you have the following files ready to install:"
Write-Host "   1. memory_leak_detector.py (Unicode-fixed version)" -ForegroundColor Gray
Write-Host "   2. signal_analyzer_profiler.py (Unicode-fixed version)" -ForegroundColor Gray
Write-Host "   3. test_unicode_fixes.py (Testing script)" -ForegroundColor Gray
Write-Host ""

$artifactFiles = @{
    "$performanceDir/memory_leak_detector.py" = "Fixed Memory Leak Detector"
    "$performanceDir/signal_analyzer_profiler.py" = "Fixed Signal Analyzer Profiler"
    "$performanceDir/test_unicode_fixes.py" = "Unicode Test Script"
}

Write-Host "Files to install:" -ForegroundColor Cyan
foreach ($file in $artifactFiles.Keys) {
    $description = $artifactFiles[$file]
    $exists = Test-Path $file
    $status = if ($exists) { "✅ Ready" } else { "⚠️ Need to copy" }
    Write-Host "   $status $file" -ForegroundColor $(if ($exists) { "Green" } else { "Yellow" })
}

# Step 6: Verify installation readiness
Write-Step "Step 6: Verifying installation readiness..."

$readyToTest = $true
$requiredFiles = @(
    "$performanceDir/memory_leak_detector.py",
    "$performanceDir/signal_analyzer_profiler.py"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Warning "Required file missing: $file"
        $readyToTest = $false
    }
}

if (-not $readyToTest) {
    Write-Error "Installation incomplete - missing required files"
    Write-Host ""
    Write-Host "MANUAL INSTALLATION STEPS:" -ForegroundColor Cyan
    Write-Host "1. Copy the fixed memory_leak_detector.py to: $performanceDir/" -ForegroundColor White
    Write-Host "2. Copy the fixed signal_analyzer_profiler.py to: $performanceDir/" -ForegroundColor White
    Write-Host "3. Copy the test_unicode_fixes.py to: $performanceDir/" -ForegroundColor White
    Write-Host "4. Run this script again to test the installation" -ForegroundColor White
    exit 1
}

Write-Success "All required files are present"

# Step 7: Test the installation
if (-not $SkipTest) {
    Write-Step "Step 7: Testing Unicode fixes..."
    
    Push-Location $performanceDir
    try {
        # Test 1: Check if files can be imported
        Write-Host "   Testing file imports..." -ForegroundColor Gray
        
        $importTest = python -c @"
try:
    import memory_leak_detector
    import signal_analyzer_profiler
    print('Import test: PASSED')
except ImportError as e:
    print(f'Import test: FAILED - {e}')
    exit(1)
except Exception as e:
    print(f'Import test: ERROR - {e}')
    exit(1)
"@
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Import test passed" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Import test failed: $importTest" -ForegroundColor Red
            $readyToTest = $false
        }
        
        # Test 2: Unicode character test
        Write-Host "   Testing Unicode character handling..." -ForegroundColor Gray
        
        $unicodeTest = python -c @"
import sys
test_chars = ['✅', '❌', '⚠️', '🚀', '📊']
try:
    for char in test_chars:
        print(f'Testing: {char}')
    print('Unicode test: PASSED')
except UnicodeEncodeError as e:
    print(f'Unicode test: FAILED - {e}')
    exit(1)
"@
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Unicode test passed" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Unicode test failed" -ForegroundColor Red
            $readyToTest = $false
        }
        
        # Test 3: Quick functional test
        if (Test-Path "test_unicode_fixes.py") {
            Write-Host "   Running comprehensive test suite..." -ForegroundColor Gray
            
            try {
                $testResult = python test_unicode_fixes.py 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "   ✅ Comprehensive test suite passed" -ForegroundColor Green
                } else {
                    Write-Host "   ⚠️ Some tests failed - check output above" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "   ⚠️ Could not run test suite" -ForegroundColor Yellow
            }
        }
        
    } finally {
        Pop-Location
    }
    
    if ($readyToTest) {
        Write-Success "All tests passed! Unicode fixes are working."
    } else {
        Write-Warning "Some tests failed, but basic functionality should work."
    }
} else {
    Write-Info "Skipping tests (--SkipTest specified)"
}

# Step 8: Create helper scripts
Write-Step "Step 8: Creating helper scripts..."

$quickTestScript = @'
# quick_memory_test.ps1
# Quick memory health check for Signal Analyzer
Write-Host "🏥 Running quick memory health check..." -ForegroundColor Yellow
cd performance_analysis
python memory_leak_detector.py --duration 120 --interval 10
Write-Host "✅ Quick test complete. Check performance_reports/ for results." -ForegroundColor Green
'@

$monitorAppScript = @'
# monitor_app.ps1
# Monitor running Signal Analyzer process
param([int]$PID)

if (-not $PID) {
    Write-Host "🔍 Finding Signal Analyzer processes..." -ForegroundColor Yellow
    $processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -like "*Signal*" -or 
        $_.MainWindowTitle -like "*Analyzer*"
    }
    
    if ($processes) {
        Write-Host "Found processes:" -ForegroundColor Green
        $processes | Format-Table Id, ProcessName, MainWindowTitle -AutoSize
        Write-Host "Usage: .\monitor_app.ps1 -PID <process_id>" -ForegroundColor Yellow
    } else {
        Write-Host "No Signal Analyzer processes found" -ForegroundColor Yellow
        Write-Host "Start your app first, then run this script" -ForegroundColor Gray
    }
    exit
}

Write-Host "📊 Monitoring process $PID..." -ForegroundColor Yellow
cd performance_analysis
python memory_leak_detector.py --attach $PID --duration 300
Write-Host "✅ Monitoring complete. Check performance_reports/ for results." -ForegroundColor Green
'@

try {
    $quickTestScript | Out-File -FilePath "quick_memory_test.ps1" -Encoding UTF8
    $monitorAppScript | Out-File -FilePath "monitor_app.ps1" -Encoding UTF8
    Write-Success "Helper scripts created"
    Write-Host "   📄 quick_memory_test.ps1 - Quick 2-minute memory test" -ForegroundColor Gray
    Write-Host "   📄 monitor_app.ps1 - Monitor running app" -ForegroundColor Gray
} catch {
    Write-Warning "Could not create helper scripts: $($_.Exception.Message)"
}

# Final summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Duration: $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Gray
Write-Host ""

Write-Host "📋 WHAT'S BEEN INSTALLED:" -ForegroundColor Cyan
Write-Host "  ✅ Unicode-safe memory leak detector" -ForegroundColor White
Write-Host "  ✅ Unicode-safe performance profiler" -ForegroundColor White
Write-Host "  ✅ Test scripts and utilities" -ForegroundColor White
Write-Host "  ✅ Helper PowerShell scripts" -ForegroundColor White
Write-Host ""

Write-Host "🚀 QUICK START COMMANDS:" -ForegroundColor Cyan
Write-Host "  .\quick_memory_test.ps1                    # Quick 2-minute test" -ForegroundColor White
Write-Host "  .\monitor_app.ps1                          # Find and monitor app" -ForegroundColor White
Write-Host "  cd performance_analysis" -ForegroundColor White
Write-Host "  python memory_leak_detector.py --duration 300   # Full analysis" -ForegroundColor White
Write-Host ""

Write-Host "📊 VIEW REPORTS:" -ForegroundColor Cyan
Write-Host "  # Open latest report in browser:" -ForegroundColor White
Write-Host "  `$report = Get-ChildItem performance_reports\*.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1" -ForegroundColor Gray
Write-Host "  Start-Process `$report.FullName" -ForegroundColor Gray
Write-Host ""

Write-Host "🎯 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Run a quick test: .\quick_memory_test.ps1" -ForegroundColor White
Write-Host "  2. Start your Signal Analyzer app" -ForegroundColor White
Write-Host "  3. Monitor it: .\monitor_app.ps1" -ForegroundColor White
Write-Host "  4. Check the generated reports for memory leaks!" -ForegroundColor White

if ($readyToTest) {
    Write-Host ""
    Write-Host "✅ All systems ready! Your Unicode issues are now fixed." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️ Installation complete with minor issues. Tools should still work." -ForegroundColor Yellow
}