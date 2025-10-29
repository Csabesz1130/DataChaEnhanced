# Heroku Deployment Script for Signal Analyzer (PowerShell)
# This script automates the deployment process

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Signal Analyzer - Heroku Deployment" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Function to print colored output
function Print-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Print-Info {
    param([string]$Message)
    Write-Host "➤ $Message" -ForegroundColor Yellow
}

# Check if Heroku CLI is installed
$herokuFound = $false
try {
    $null = Get-Command heroku -ErrorAction Stop
    $herokuFound = $true
} catch {
    Print-Info "Heroku CLI not found in PATH, checking common installation locations..."
    
    # Common Heroku CLI installation paths
    $possiblePaths = @(
        "C:\Program Files\heroku\bin",
        "${env:LOCALAPPDATA}\heroku\bin",
        "${env:PROGRAMFILES}\heroku\bin",
        "${env:PROGRAMFILES(X86)}\heroku\bin"
    )
    
    foreach ($path in $possiblePaths) {
        $herokuCmd = Join-Path $path "heroku.cmd"
        if (Test-Path $herokuCmd) {
            Print-Info "Found Heroku CLI at: $path"
            Print-Info "Adding to current session PATH..."
            $env:Path = $env:Path + ";" + $path
            $herokuFound = $true
            break
        }
    }
}

if (-not $herokuFound) {
    Print-Error "Heroku CLI not found!"
    Write-Host "Install from: https://devcenter.heroku.com/articles/heroku-cli"
    Write-Host "After installation, restart your terminal or Cursor."
    exit 1
}

Print-Success "Heroku CLI found"

# Verify Heroku is working
try {
    $herokuVersion = heroku --version 2>&1 | Out-String
    Print-Success "Heroku CLI is working"
} catch {
    Print-Error "Failed to execute Heroku CLI"
    exit 1
}

# Check if logged in to Heroku
try {
    $username = heroku auth:whoami 2>&1 | Out-String
    if ($username -match "not logged in") {
        throw "Not logged in"
    }
    Print-Success "Logged in to Heroku as: $($username.Trim())"
} catch {
    Print-Info "Logging in to Heroku..."
    heroku login
    $username = heroku auth:whoami
    Print-Success "Logged in as: $($username.Trim())"
}

# Get app name
Print-Info "Enter your Heroku app name (or press Enter to use 'datachaenhanced'):"
$APP_NAME = Read-Host
if ([string]::IsNullOrWhiteSpace($APP_NAME)) {
    $APP_NAME = "datachaenhanced"
}

# Check if app exists
$appExists = $false
try {
    $appInfo = heroku apps:info --app $APP_NAME 2>&1
    if ($LASTEXITCODE -eq 0) {
        $appExists = $true
        Print-Success "Using existing app: $APP_NAME"
    }
} catch {
    # App doesn't exist
}

if (-not $appExists) {
    Print-Info "App doesn't exist. Creating..."
    heroku create $APP_NAME
    Print-Success "Created app: $APP_NAME"
}

Write-Host ""
Print-Info "App URL: https://$APP_NAME.herokuapp.com"
Write-Host ""

# Add PostgreSQL addon if not exists
Print-Info "Checking for PostgreSQL addon..."
$addons = heroku addons --app $APP_NAME 2>&1 | Out-String
if ($addons -notmatch "heroku-postgresql") {
    Print-Info "Adding PostgreSQL addon..."
    try {
        heroku addons:create heroku-postgresql:essential-0 --app $APP_NAME
        Print-Success "PostgreSQL addon added"
    } catch {
        Print-Info "Note: Trying to add PostgreSQL addon..."
    }
} else {
    Print-Success "PostgreSQL addon already exists"
}

# Set environment variables
Print-Info "Setting environment variables..."

# Generate secret key if not set
$existingSecret = heroku config:get SECRET_KEY --app $APP_NAME 2>&1 | Out-String
if ([string]::IsNullOrWhiteSpace($existingSecret.Trim())) {
    # Generate random secret key
    $SECRET_KEY = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 64 | ForEach-Object {[char]$_})
    heroku config:set "SECRET_KEY=$SECRET_KEY" --app $APP_NAME
    Print-Success "SECRET_KEY set"
} else {
    Print-Success "SECRET_KEY already exists"
}

heroku config:set FLASK_ENV=production --app $APP_NAME
heroku config:set MAX_UPLOAD_SIZE=52428800 --app $APP_NAME
heroku config:set "FRONTEND_URL=https://$APP_NAME.herokuapp.com" --app $APP_NAME

Print-Success "Environment variables configured"

# Build frontend
Print-Info "Building frontend..."
if (Test-Path "frontend") {
    Push-Location frontend
    
    if (-not (Test-Path "node_modules")) {
        Print-Info "Installing frontend dependencies..."
        npm install
    }
    
    Print-Info "Building React app..."
    npm run build
    Pop-Location
    Print-Success "Frontend built successfully"
} else {
    Print-Error "Frontend directory not found!"
    exit 1
}

# Check current branch
$currentBranch = git branch --show-current
Print-Info "Current branch: $currentBranch"

# Commit changes
Print-Info "Committing changes..."
git add .
$commitDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMsg = "Deploy to Heroku - $commitDate"
git commit -m $commitMsg 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Print-Success "Changes committed"
} else {
    Print-Info "No changes to commit"
}

# Add Heroku remote if not exists
$remotes = git remote
if ($remotes -notcontains "heroku") {
    heroku git:remote --app $APP_NAME
    Print-Success "Heroku remote added"
} else {
    Print-Success "Heroku remote already exists"
}

# Deploy to Heroku
Print-Info "Deploying to Heroku..."
Write-Host ""

# Determine which branch to push
if ($currentBranch -ne "main") {
    Print-Info "You're on branch '$currentBranch', not 'main'"
    $response = Read-Host "Push '$currentBranch' to Heroku? (Y/n)"
    if ($response -ne "n" -and $response -ne "N") {
        git push heroku "${currentBranch}:main"
    } else {
        Print-Info "Switching to main branch..."
        git checkout main
        git push heroku main
    }
} else {
    git push heroku main
}

if ($LASTEXITCODE -eq 0) {
    Print-Success "Deployment successful!"
} else {
    Print-Error "Deployment failed!"
    Print-Info "Check logs with: heroku logs --tail --app $APP_NAME"
    exit 1
}

# Initialize database
Print-Info "Initializing database..."
$dbInit = 'python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"'
heroku run $dbInit --app $APP_NAME
Print-Success "Database initialization attempted"

# Display completion message
Write-Host ""
Print-Success "======================================"
Print-Success "Deployment Complete!"
Print-Success "======================================"
Write-Host ""
Print-Info "Your app is live at: https://$APP_NAME.herokuapp.com"
Write-Host ""
Print-Info "Useful commands:"
Write-Host "  heroku logs --tail --app $APP_NAME        # View logs" -ForegroundColor Cyan
Write-Host "  heroku ps --app $APP_NAME                 # Check dyno status" -ForegroundColor Cyan
Write-Host "  heroku config --app $APP_NAME             # View config vars" -ForegroundColor Cyan
Write-Host "  heroku pg:psql --app $APP_NAME            # Connect to database" -ForegroundColor Cyan
Write-Host "  heroku restart --app $APP_NAME            # Restart app" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to open the app
$openResponse = Read-Host "Open app in browser? (Y/n)"
if ($openResponse -ne "n" -and $openResponse -ne "N") {
    heroku open --app $APP_NAME
}

Print-Success "Done!"
