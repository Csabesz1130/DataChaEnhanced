#!/bin/bash

# Heroku Deployment Script for Signal Analyzer
# This script automates the deployment process

set -e  # Exit on error

echo "======================================"
echo "Signal Analyzer - Heroku Deployment"
echo "======================================"
echo

# Colors for output
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➤ $1${NC}"
}

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    print_error "Heroku CLI not found!"
    echo "Install from: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi
print_success "Heroku CLI found"

# Check if logged in to Heroku
if ! heroku auth:whoami &> /dev/null; then
    print_info "Logging in to Heroku..."
    heroku login
fi
print_success "Logged in to Heroku"

# Get app name
print_info "Enter your Heroku app name (or press Enter to create new):"
read -r APP_NAME

if [ -z "$APP_NAME" ]; then
    print_info "Creating new Heroku app..."
    APP_NAME=$(heroku create | grep -oP 'https://\K[^.]+')
    print_success "Created app: $APP_NAME"
else
    # Check if app exists
    if heroku apps:info --app "$APP_NAME" &> /dev/null; then
        print_success "Using existing app: $APP_NAME"
    else
        print_info "App doesn't exist. Creating..."
        heroku create "$APP_NAME"
        print_success "Created app: $APP_NAME"
    fi
fi

echo
print_info "App URL: https://$APP_NAME.herokuapp.com"
echo

# Add PostgreSQL addon if not exists
print_info "Checking for PostgreSQL addon..."
if ! heroku addons --app "$APP_NAME" | grep -q heroku-postgresql; then
    print_info "Adding PostgreSQL addon..."
    heroku addons:create heroku-postgresql:mini --app "$APP_NAME"
    print_success "PostgreSQL addon added"
else
    print_success "PostgreSQL addon already exists"
fi

# Add Redis addon (optional)
print_info "Add Redis addon? (y/N):"
read -r ADD_REDIS
if [ "$ADD_REDIS" = "y" ] || [ "$ADD_REDIS" = "Y" ]; then
    if ! heroku addons --app "$APP_NAME" | grep -q heroku-redis; then
        heroku addons:create heroku-redis:mini --app "$APP_NAME"
        print_success "Redis addon added"
    else
        print_success "Redis addon already exists"
    fi
fi

# Set environment variables
print_info "Setting environment variables..."

# Generate secret key if not set
if ! heroku config:get SECRET_KEY --app "$APP_NAME" | grep -q "."; then
    SECRET_KEY=$(openssl rand -hex 32)
    heroku config:set SECRET_KEY="$SECRET_KEY" --app "$APP_NAME"
    print_success "SECRET_KEY set"
fi

heroku config:set FLASK_ENV=production --app "$APP_NAME"
heroku config:set MAX_UPLOAD_SIZE=52428800 --app "$APP_NAME"
heroku config:set FRONTEND_URL="https://$APP_NAME.herokuapp.com" --app "$APP_NAME"

print_success "Environment variables configured"

# Build frontend
print_info "Building frontend..."
if [ -d "frontend" ]; then
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        print_info "Installing frontend dependencies..."
        npm install
    fi
    
    print_info "Building React app..."
    npm run build
    cd ..
    print_success "Frontend built successfully"
else
    print_error "Frontend directory not found!"
    exit 1
fi

# Commit changes
print_info "Committing changes..."
git add .
if git commit -m "Deploy to Heroku - $(date +%Y-%m-%d\ %H:%M:%S)"; then
    print_success "Changes committed"
else
    print_info "No changes to commit"
fi

# Add Heroku remote if not exists
if ! git remote | grep -q heroku; then
    heroku git:remote --app "$APP_NAME"
    print_success "Heroku remote added"
fi

# Deploy to Heroku
print_info "Deploying to Heroku..."
echo
if git push heroku main; then
    print_success "Deployment successful!"
else
    print_error "Deployment failed!"
    print_info "Check logs with: heroku logs --tail --app $APP_NAME"
    exit 1
fi

# Initialize database
print_info "Initializing database..."
heroku run "python -c \"from backend.app import app, db; app.app_context().push(); db.create_all()\"" --app "$APP_NAME"
print_success "Database initialized"

# Run database migrations if needed
# heroku run python backend/migrations.py --app "$APP_NAME"

# Open app
echo
print_success "======================================"
print_success "Deployment Complete!"
print_success "======================================"
echo
print_info "Your app is live at: https://$APP_NAME.herokuapp.com"
echo
print_info "Useful commands:"
echo "  heroku logs --tail --app $APP_NAME        # View logs"
echo "  heroku ps --app $APP_NAME                 # Check dyno status"
echo "  heroku config --app $APP_NAME             # View config vars"
echo "  heroku pg:psql --app $APP_NAME            # Connect to database"
echo "  heroku restart --app $APP_NAME            # Restart app"
echo

# Ask if user wants to open the app
print_info "Open app in browser? (Y/n):"
read -r OPEN_APP
if [ "$OPEN_APP" != "n" ] && [ "$OPEN_APP" != "N" ]; then
    heroku open --app "$APP_NAME"
fi

print_success "Done!"

