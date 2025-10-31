# Heroku Deployment Script for Signal Analyzer (PowerShell)
# This script automates the deployment process for Windows

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Signal Analyzer - Heroku Deployment" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Function to find Heroku CLI
function Find-Heroku {
    # Check if heroku is in PATH
    if (Get-Command heroku -ErrorAction SilentlyContinue) {
        return "heroku"
    }
    
    # Common Heroku installation paths on Windows
    $herokuPaths = @(
        "C:\Program Files\heroku\bin\heroku.exe"
        "C:\Program Files (x86)\heroku\bin\heroku.exe"
        "$env:LOCALAPPDATA\Programs\heroku\bin\heroku.exe"
        "$env:USERPROFILE\AppData\Local\Programs\heroku\bin\heroku.exe"
    )
    
    foreach ($path in $herokuPaths) {
        if (Test-Path $path) {
            $herokuDir = Split-Path $path -Parent
            if ($env:Path -notlike "*$herokuDir*") {
                $env:Path += ";$herokuDir"
            }
            
            # Verify it works
            if (Get-Command heroku -ErrorAction SilentlyContinue) {
                Write-Host "Found Heroku at: $path" -ForegroundColor Yellow
                return "heroku"
            }
        }
    }
    
    return $null
}

# Check if Heroku CLI is installed
Write-Host "Checking for Heroku CLI..." -ForegroundColor Yellow
$herokuCmd = Find-Heroku

if (-not $herokuCmd) {
    Write-Host "Heroku CLI not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Heroku CLI from: https://devcenter.heroku.com/articles/heroku-cli" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installation, you may need to:"
    Write-Host "  1. Restart your terminal"
    Write-Host "  2. Or manually add Heroku to your PATH"
    Write-Host ""
    Write-Host "Common installation location:"
    Write-Host "  C:\Program Files\heroku\bin"
    Write-Host ""
    $continue = Read-Host "Do you want to continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
} else {
    Write-Host "Heroku CLI found" -ForegroundColor Green
    # Verify it works
    try {
        $version = & heroku --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Heroku CLI found but not working properly" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "Heroku CLI found but not working properly" -ForegroundColor Red
        exit 1
    }
}

# Check if logged in to Heroku
try {
    $whoami = & heroku auth:whoami 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Logging in to Heroku..." -ForegroundColor Yellow
        heroku login
    }
    Write-Host "Logged in to Heroku" -ForegroundColor Green
} catch {
    Write-Host "Error checking Heroku login status" -ForegroundColor Red
    exit 1
}

# Get app name
$appName = Read-Host "Enter your Heroku app name (or press Enter to create new)"

if ([string]::IsNullOrWhiteSpace($appName)) {
    Write-Host "Creating new Heroku app..." -ForegroundColor Yellow
    $createOutput = heroku create 2>&1
    if ($LASTEXITCODE -eq 0) {
        # Extract app name from output
        if ($createOutput -match 'https://([^.]+)\.herokuapp\.com') {
            $appName = $matches[1]
            Write-Host "Created app: $appName" -ForegroundColor Green
        } else {
            Write-Host "Failed to extract app name from output" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Failed to create Heroku app" -ForegroundColor Red
        exit 1
    }
} else {
    # Check if app exists
    $appInfo = heroku apps:info --app $appName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Using existing app: $appName" -ForegroundColor Green
    } else {
        Write-Host "App doesn't exist. Creating..." -ForegroundColor Yellow
        heroku create $appName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Created app: $appName" -ForegroundColor Green
        } else {
            Write-Host "Failed to create app" -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""
Write-Host "App URL: https://$appName.herokuapp.com" -ForegroundColor Yellow
Write-Host ""

# Add PostgreSQL addon if not exists
Write-Host "Checking for PostgreSQL addon..." -ForegroundColor Yellow
$addons = heroku addons --app $appName 2>&1
if ($addons -notmatch 'heroku-postgresql') {
    Write-Host "Adding PostgreSQL addon..." -ForegroundColor Yellow
    heroku addons:create heroku-postgresql:mini --app $appName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL addon added" -ForegroundColor Green
    } else {
        Write-Host "Warning: Failed to add PostgreSQL addon" -ForegroundColor Yellow
    }
} else {
    Write-Host "PostgreSQL addon already exists" -ForegroundColor Green
}

# Add Redis addon (optional)
$addRedis = Read-Host "Add Redis addon? (y/N)"
if ($addRedis -eq "y" -or $addRedis -eq "Y") {
    if ($addons -notmatch 'heroku-redis') {
        heroku addons:create heroku-redis:mini --app $appName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Redis addon added" -ForegroundColor Green
        }
    } else {
        Write-Host "Redis addon already exists" -ForegroundColor Green
    }
}

# Set environment variables
Write-Host "Setting environment variables..." -ForegroundColor Yellow

# Generate secret key if not set
$secretKey = heroku config:get SECRET_KEY --app $appName 2>&1
if ([string]::IsNullOrWhiteSpace($secretKey) -or $LASTEXITCODE -ne 0) {
    # Generate random secret key
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $secretKey = [System.BitConverter]::ToString($bytes).Replace("-", "").ToLower()
    heroku config:set SECRET_KEY="$secretKey" --app $appName
    Write-Host "SECRET_KEY set" -ForegroundColor Green
}

heroku config:set FLASK_ENV=production --app $appName
heroku config:set MAX_UPLOAD_SIZE=52428800 --app $appName
heroku config:set FRONTEND_URL="https://$appName.herokuapp.com" --app $appName

Write-Host "Environment variables configured" -ForegroundColor Green

# Build frontend
Write-Host "Building frontend..." -ForegroundColor Yellow
if (Test-Path "frontend") {
    Push-Location frontend
    
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    Write-Host "Building React app..." -ForegroundColor Yellow
    npm run build
    
    Pop-Location
    Write-Host "Frontend built successfully" -ForegroundColor Green
} else {
    Write-Host "Frontend directory not found!" -ForegroundColor Red
    exit 1
}

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
git add .
$commitMessage = "Deploy to Heroku - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
git commit -m $commitMessage
if ($LASTEXITCODE -ne 0) {
    Write-Host "No changes to commit" -ForegroundColor Yellow
} else {
    Write-Host "Changes committed" -ForegroundColor Green
}

# Add Heroku remote if not exists
$remotes = git remote 2>&1
if ($remotes -notmatch 'heroku') {
    heroku git:remote --app $appName
    Write-Host "Heroku remote added" -ForegroundColor Green
}

# Deploy to Heroku
Write-Host "Deploying to Heroku..." -ForegroundColor Yellow
Write-Host ""
git push heroku main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful!" -ForegroundColor Green
} else {
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "Check logs with: heroku logs --tail --app $appName" -ForegroundColor Yellow
    exit 1
}

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Yellow
heroku run "python -c `"from backend.app import app, db; app.app_context().push(); db.create_all()`"" --app $appName
Write-Host "Database initialized" -ForegroundColor Green

# Open app
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your app is live at: https://$appName.herokuapp.com" -ForegroundColor Yellow
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  heroku logs --tail --app $appName        # View logs"
Write-Host "  heroku ps --app $appName                 # Check dyno status"
Write-Host "  heroku config --app $appName             # View config vars"
Write-Host "  heroku pg:psql --app $appName            # Connect to database"
Write-Host "  heroku restart --app $appName            # Restart app"
Write-Host ""

# Ask if user wants to open the app
$openApp = Read-Host "Open app in browser? (Y/n)"
if ($openApp -ne "n" -and $openApp -ne "N") {
    heroku open --app $appName
}

Write-Host "Done!" -ForegroundColor Green
