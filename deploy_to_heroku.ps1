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
    $createOutput = heroku create 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        # Extract app name from output - try multiple patterns
        $appName = $null
        if ($createOutput -match 'done, [\p{Sc}\p{So}]?\s*([a-z0-9-]+)') {
            $appName = $matches[1]
        } elseif ($createOutput -match 'Creating app\.?\s*([a-z0-9-]+)') {
            $appName = $matches[1]
        } elseif ($createOutput -match 'https://([a-z0-9-]+)\.herokuapp\.com') {
            $appName = $matches[1]
        } elseif ($createOutput -match '\| ([a-z0-9-]+) \|') {
            $appName = $matches[1]
        }

        if ($appName -match '^([a-z0-9-]+)-[a-f0-9]{10,}$') {
            $appName = $matches[1]
        }
        
        if ([string]::IsNullOrWhiteSpace($appName)) {
            Write-Host "Failed to extract app name from output" -ForegroundColor Red
            Write-Host "Output was: $createOutput" -ForegroundColor Yellow
            Write-Host "Please enter the app name manually:" -ForegroundColor Yellow
            $appName = Read-Host "App name"
            if ([string]::IsNullOrWhiteSpace($appName)) {
                Write-Host "App name is required" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Created app: $appName" -ForegroundColor Green
        }
    } else {
        Write-Host "Failed to create Heroku app" -ForegroundColor Red
        Write-Host "Output: $createOutput" -ForegroundColor Yellow
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

# Validate app name before proceeding
if ([string]::IsNullOrWhiteSpace($appName)) {
    Write-Host "Error: App name is empty! Cannot proceed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "App URL: https://$appName.herokuapp.com" -ForegroundColor Yellow
Write-Host ""

# Add PostgreSQL addon if not exists
Write-Host "Checking for PostgreSQL addon..." -ForegroundColor Yellow
$addons = heroku addons --app $appName 2>&1
if ($addons -notmatch 'heroku-postgresql') {
    Write-Host "Adding PostgreSQL addon..." -ForegroundColor Yellow
    Write-Host "Note: Using 'essential-0' plan (mini plan is end-of-life)" -ForegroundColor Yellow
    # Try essential-0 plan first (free tier alternative)
    heroku addons:create heroku-postgresql:essential-0 --app $appName
    if ($LASTEXITCODE -ne 0) {
        # If that fails, try to get available plans
        Write-Host "Failed to add essential-0 plan. Checking available plans..." -ForegroundColor Yellow
        heroku addons:plans heroku-postgresql 2>&1 | Out-String
        Write-Host "Please manually add PostgreSQL addon with: heroku addons:create heroku-postgresql:PLAN_NAME --app $appName" -ForegroundColor Yellow
    } else {
        Write-Host "PostgreSQL addon added" -ForegroundColor Green
    }
} else {
    Write-Host "PostgreSQL addon already exists" -ForegroundColor Green
}

# Add Redis addon (optional)
$addRedis = Read-Host "Add Redis addon? (y/N)"
if ($addRedis -eq "y" -or $addRedis -eq "Y") {
    # Refresh addons list
    $addons = heroku addons --app $appName 2>&1
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
    # Generate random secret key (compatible with older PowerShell versions)
    $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::Create()
    $bytes = New-Object byte[] 32
    $rng.GetBytes($bytes)
    $secretKey = [System.BitConverter]::ToString($bytes).Replace("-", "").ToLower()
    $rng.Dispose()
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

# Validate app name before proceeding
if ([string]::IsNullOrWhiteSpace($appName)) {
    Write-Host "Error: App name is empty!" -ForegroundColor Red
    exit 1
}

# Ensure Heroku remote points to the selected app
Write-Host "Configuring Heroku git remote..." -ForegroundColor Yellow
heroku git:remote --app $appName
if ($LASTEXITCODE -eq 0) {
    Write-Host "Heroku remote configured" -ForegroundColor Green
} else {
    Write-Host "Failed to configure Heroku remote" -ForegroundColor Red
    exit 1
}

# Deploy to Heroku
Write-Host "Deploying to Heroku..." -ForegroundColor Yellow
Write-Host ""

# Determine the ref we will push
$currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    $currentBranch = "HEAD"
}

$pushSource = if ($currentBranch -eq "HEAD") { "HEAD" } else { $currentBranch }
$commitHash = (git rev-parse HEAD).Trim()

if ($pushSource -eq "HEAD") {
    Write-Host "Detached HEAD detected; pushing commit $commitHash to Heroku main..." -ForegroundColor Yellow
} else {
    Write-Host "Pushing branch $pushSource (commit $commitHash) to Heroku main..." -ForegroundColor Yellow
}

git push heroku "$pushSource:main" --force-with-lease

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful!" -ForegroundColor Green
} else {
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "Check logs with: heroku logs --tail --app $appName" -ForegroundColor Yellow
    exit 1
}

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Yellow
# Use proper escaping for PowerShell - escape quotes properly
$pythonCmd = 'python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"'
heroku run --app $appName -- $pythonCmd
if ($LASTEXITCODE -eq 0) {
    Write-Host "Database initialized" -ForegroundColor Green
} else {
    Write-Host "Warning: Database initialization may have failed" -ForegroundColor Yellow
}

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
with the todos