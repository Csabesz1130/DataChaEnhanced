# Start Backend Flask Server
Write-Host "Starting Flask Backend Server..." -ForegroundColor Green

# Set environment variables
$env:FLASK_APP = "backend/app.py"
$env:FLASK_ENV = "development"
$env:DATABASE_URL = "sqlite:///test.db"

# Navigate to project root
Set-Location $PSScriptRoot

# Start Flask server
Write-Host "Backend will start on http://localhost:5000" -ForegroundColor Yellow
python backend/app.py

