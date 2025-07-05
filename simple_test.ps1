# simple_test.ps1
param([switch]$SkipTest)

Clear-Host
Write-Host "PERFORMANCE TOOLS TEST" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

$errors = 0

# Test 1: Check files exist
Write-Host "1. Checking files..." -ForegroundColor Yellow
if (Test-Path "performance_analysis\memory_leak_detector.py") {
    Write-Host "   memory_leak_detector.py - OK" -ForegroundColor Green
} else {
    Write-Host "   memory_leak_detector.py - MISSING" -ForegroundColor Red
    $errors++
}

if (Test-Path "performance_analysis\signal_analyzer_profiler.py") {
    Write-Host "   signal_analyzer_profiler.py - OK" -ForegroundColor Green
} else {
    Write-Host "   signal_analyzer_profiler.py - MISSING" -ForegroundColor Red
    $errors++
}

# Test 2: Check Python dependencies
Write-Host "2. Checking dependencies..." -ForegroundColor Yellow
$deps = @("psutil", "matplotlib", "numpy")
$missingDeps = @()

foreach ($dep in $deps) {
    try {
        $result = python -c "import $dep; print('OK')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   $dep - OK" -ForegroundColor Green
        } else {
            Write-Host "   $dep - MISSING" -ForegroundColor Red
            $missingDeps += $dep
        }
    } catch {
        Write-Host "   $dep - ERROR" -ForegroundColor Red
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -eq 0) {
    Write-Host "ALL DEPENDENCIES OK!" -ForegroundColor Green
} else {
    Write-Host "Install missing: pip install $($missingDeps -join ' ')" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Ready to monitor your Signal Analyzer app!" -ForegroundColor Green
