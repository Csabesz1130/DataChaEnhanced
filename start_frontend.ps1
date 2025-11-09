# Start Frontend React Server
Write-Host "Starting React Frontend..." -ForegroundColor Green

# Navigate to frontend directory
Set-Location $PSScriptRoot\frontend

# Start React development server
Write-Host "Frontend will start on http://localhost:3000" -ForegroundColor Yellow
npm start

