# Quick Repository Analyzer - PowerShell
# Fast analysis of repository size issues

Write-Host "Quick Repository Size Check" -ForegroundColor Cyan
Write-Host ("=" * 40) -ForegroundColor Blue

$totalSavings = 0

# Function to get directory size safely with timeout
function Get-DirectorySizeFast {
    param([string]$Path)
    try {
        $job = Start-Job -ScriptBlock {
            param($dir)
            (Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue | 
             Measure-Object -Property Length -Sum).Sum
        } -ArgumentList $Path
        
        $result = Wait-Job $job -Timeout 10 | Receive-Job
        Remove-Job $job -Force
        return $result
    }
    catch {
        return 0
    }
}

# Function to format size
function Format-Size {
    param([long]$Size)
    if ($Size -gt 1GB) { return "{0:N1} GB" -f ($Size / 1GB) }
    elseif ($Size -gt 1MB) { return "{0:N1} MB" -f ($Size / 1MB) }
    elseif ($Size -gt 1KB) { return "{0:N1} KB" -f ($Size / 1KB) }
    else { return "$Size B" }
}

Write-Host ""
Write-Host "Checking for common space wasters..." -ForegroundColor Yellow

# Quick check for obvious space wasters
$spaceWasters = @(
    @{Name="node_modules"; Description="Node.js dependencies"},
    @{Name="__pycache__"; Description="Python cache"},
    @{Name="build"; Description="Build artifacts"},
    @{Name="dist"; Description="Distribution files"},
    @{Name=".vscode"; Description="VS Code settings"},
    @{Name=".idea"; Description="IntelliJ settings"},
    @{Name="venv"; Description="Python virtual environment"},
    @{Name="env"; Description="Environment directory"},
    @{Name="target"; Description="Build target"},
    @{Name="bower_components"; Description="Bower dependencies"},
    @{Name="vendor"; Description="Vendor dependencies"},
    @{Name=".git"; Description="Git repository data"}
)

$foundIssues = @()

foreach ($waster in $spaceWasters) {
    Write-Host "  Checking for $($waster.Name)..." -ForegroundColor Gray
    
    try {
        $items = Get-ChildItem -Path . -Name $waster.Name -Directory -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            if (Test-Path $item) {
                Write-Host "    Found: $item" -ForegroundColor Yellow
                
                # Quick size estimation
                $size = Get-DirectorySizeFast -Path $item
                if ($size -eq $null) { $size = 0 }
                
                if ($size -gt 1MB) {
                    $foundIssues += @{
                        Path = $item
                        Size = $size
                        Description = $waster.Description
                        Type = "Directory"
                    }
                    $script:totalSavings += $size
                    
                    Write-Host "      Size: $(Format-Size $size)" -ForegroundColor Red
                }
            }
        }
    }
    catch {
        Write-Host "    Error checking $($waster.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Check for large files in root directory
Write-Host ""
Write-Host "Checking for large files in root..." -ForegroundColor Yellow
try {
    $largeFiles = Get-ChildItem -Path . -File | Where-Object { $_.Length -gt 10MB }
    foreach ($file in $largeFiles) {
        $foundIssues += @{
            Path = $file.Name
            Size = $file.Length
            Description = "Large file"
            Type = "File"
        }
        $script:totalSavings += $file.Length
        Write-Host "  Large file: $($file.Name) - $(Format-Size $file.Length)" -ForegroundColor Red
    }
}
catch {
    Write-Host "  Error checking large files: $($_.Exception.Message)" -ForegroundColor Red
}

# Results
if ($foundIssues.Count -eq 0) {
    Write-Host ""
    Write-Host "No major space issues found in quick scan!" -ForegroundColor Green
    Write-Host "Your repository appears to be clean." -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "Found $($foundIssues.Count) space issues:" -ForegroundColor Yellow
    Write-Host ("-" * 60) -ForegroundColor Blue
    
    $sortedIssues = $foundIssues | Sort-Object { $_.Size } -Descending
    for ($i = 0; $i -lt [Math]::Min(10, $sortedIssues.Count); $i++) {
        $issue = $sortedIssues[$i]
        $icon = if ($issue.Type -eq "Directory") { "[DIR]" } else { "[FILE]" }
        Write-Host "$($i+1). $icon $(Format-Size $issue.Size) - $($issue.Path)" -ForegroundColor White
        Write-Host "   $($issue.Description)" -ForegroundColor Gray
    }
    
    Write-Host ("-" * 60) -ForegroundColor Blue
    Write-Host "Total potential savings: $(Format-Size $totalSavings)" -ForegroundColor Green
    
    # Generate cleanup commands
    Write-Host ""
    Write-Host "PowerShell Cleanup Commands:" -ForegroundColor Cyan
    Write-Host "Copy and paste these commands (add -WhatIf to preview first):" -ForegroundColor Yellow
    
    # Node modules
    if ($foundIssues | Where-Object { $_.Path -like "*node_modules*" }) {
        Write-Host ""
        Write-Host "# Remove Node.js dependencies:" -ForegroundColor Yellow
        Write-Host "Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue" -ForegroundColor White
        Write-Host "git rm -r --cached node_modules/ 2>`$null" -ForegroundColor White
    }
    
    # Python cache
    if ($foundIssues | Where-Object { $_.Path -like "*__pycache__*" }) {
        Write-Host ""
        Write-Host "# Remove Python cache:" -ForegroundColor Yellow
        Write-Host "Get-ChildItem -Recurse -Directory -Name '__pycache__' | Remove-Item -Recurse -Force" -ForegroundColor White
        Write-Host "Get-ChildItem -Recurse -File -Name '*.pyc' | Remove-Item -Force" -ForegroundColor White
    }
    
    # Build artifacts
    if ($foundIssues | Where-Object { $_.Path -like "*build*" -or $_.Path -like "*dist*" -or $_.Path -like "*target*" }) {
        Write-Host ""
        Write-Host "# Remove build artifacts:" -ForegroundColor Yellow
        Write-Host "Remove-Item -Recurse -Force build, dist, target -ErrorAction SilentlyContinue" -ForegroundColor White
        Write-Host "git rm -r --cached build/ dist/ target/ 2>`$null" -ForegroundColor White
    }
    
    # IDE files
    if ($foundIssues | Where-Object { $_.Path -like "*vscode*" -or $_.Path -like "*idea*" }) {
        Write-Host ""
        Write-Host "# Remove IDE files:" -ForegroundColor Yellow
        Write-Host "Remove-Item -Recurse -Force .vscode, .idea -ErrorAction SilentlyContinue" -ForegroundColor White
        Write-Host "git rm -r --cached .vscode/ .idea/ 2>`$null" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Add -WhatIf to any Remove-Item command to preview changes" -ForegroundColor White
    Write-Host "2. Run the cleanup commands above" -ForegroundColor White
    Write-Host "3. Update .gitignore to prevent future issues" -ForegroundColor White
    Write-Host "4. Run: git gc --aggressive --prune=now" -ForegroundColor White
}

# Quick .gitignore check
if (!(Test-Path ".gitignore")) {
    Write-Host ""
    Write-Host "No .gitignore found! This could be why large files are tracked." -ForegroundColor Red
    Write-Host "Create a .gitignore file to prevent future issues." -ForegroundColor Yellow
}
else {
    Write-Host ""
    Write-Host ".gitignore exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Quick scan completed!" -ForegroundColor Green
Write-Host "For detailed analysis, use the Python script with --threshold 50M for faster results." -ForegroundColor Cyan