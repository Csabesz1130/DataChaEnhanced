function Test-MemoryHealth {
    param([int]$Duration = 120)
    Push-Location performance_analysis
    try {
        python memory_leak_detector.py --duration $Duration --interval 10
    } finally {
        Pop-Location
    }
}

function Show-LatestReport {
    $report = Get-ChildItem performance_reports\memory_leak_report_*.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($report) { Start-Process $report.FullName }
}