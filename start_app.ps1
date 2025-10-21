# PowerShell script to start the application with proper virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "Virtual environment activated." -ForegroundColor Green
Write-Host "Starting Signal Analyzer with drag and drop support..." -ForegroundColor Cyan

# Run the application
python run_with_hot_reload.py

Write-Host "Application closed. Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
