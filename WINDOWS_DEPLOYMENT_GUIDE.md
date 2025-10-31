# 🪟 Windows Deployment Guide - Signal Analyzer to Heroku

## What Happened?

The `deploy_to_heroku.sh` script is a **bash script** that doesn't run in PowerShell. This guide provides **Windows-specific steps**.

---

## 📋 Prerequisites (Do These First!)

### 1. Install Heroku CLI (Required)

**Download and install:**
👉 https://devcenter.heroku.com/articles/heroku-cli

**Or install via Chocolatey:**
```powershell
choco install heroku-cli
```

**Or install via npm:**
```powershell
npm install -g heroku
```

After installation, **restart PowerShell** and verify:
```powershell
heroku --version
# Should show: heroku/x.x.x
```

---

### 2. Create Heroku Account (If you don't have one)

👉 https://signup.heroku.com/

- Sign up (free)
- Verify email
- You're ready!

---

### 3. Install Node.js (For frontend build)

👉 https://nodejs.org/ (Download LTS version)

Verify installation:
```powershell
node --version
npm --version
```

---

### 4. Ensure Git is Installed

```powershell
git --version
# Should show git version
```

If not installed: https://git-scm.com/download/win

---

## 🚀 Windows Deployment Steps

### Step 1: Login to Heroku

```powershell
heroku login
```

This will open a browser window. Log in there, then return to PowerShell.

---

### Step 2: Create Heroku App

```powershell
# Option A: Let Heroku generate a name
heroku create

# Option B: Choose your own name
heroku create signal-analyzer-your-name
```

**Note the app URL** that's displayed (e.g., `https://signal-analyzer-your-name.herokuapp.com`)

---

### Step 3: Add PostgreSQL Database

```powershell
heroku addons:create heroku-postgresql:mini
```

---

### Step 4: Set Environment Variables

```powershell
# Generate a secret key (copy this entire command)
$SECRET_KEY = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
heroku config:set SECRET_KEY=$SECRET_KEY

# Set other variables
heroku config:set FLASK_ENV=production
heroku config:set MAX_UPLOAD_SIZE=52428800

# Get your app name
$APP_NAME = heroku apps:info --json | ConvertFrom-Json | Select-Object -ExpandProperty app | Select-Object -ExpandProperty name
heroku config:set FRONTEND_URL="https://$APP_NAME.herokuapp.com"

# Verify
heroku config
```

---

### Step 5: Build Frontend

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Build production version
npm run build

# Go back to root
cd ..
```

---

### Step 6: Commit All Changes

```powershell
# Add all files
git add .

# Commit
git commit -m "Prepare for Heroku deployment"

# If already committed, that's fine - skip this step
```

---

### Step 7: Deploy to Heroku

```powershell
# Push to Heroku
git push heroku main

# If your branch is named 'master' instead of 'main':
# git push heroku master
```

**This will take 2-5 minutes.** You'll see:
- Detecting buildpack
- Installing Python dependencies
- Collecting static files
- Launching...

---

### Step 8: Initialize Database

```powershell
heroku run "python -c `"from backend.app import app, db; app.app_context().push(); db.create_all()`""
```

---

### Step 9: Open Your App!

```powershell
heroku open
```

Your app should open in a browser! 🎉

---

## 🧪 Test Your Deployment

### 1. Health Check

```powershell
# Get your app URL
$APP_URL = heroku apps:info --json | ConvertFrom-Json | Select-Object -ExpandProperty app | Select-Object -ExpandProperty web_url
Write-Host "Your app URL: $APP_URL"

# Test health endpoint
Invoke-WebRequest -Uri "$APP_URL/api/health" | Select-Object -ExpandProperty Content
# Should show: {"status":"healthy",...}
```

### 2. Full Test

1. Open your app URL in browser
2. Try uploading a `.atf` file
3. Set parameters
4. Click "Run Analysis"
5. View plots
6. Export to Excel/CSV

---

## 📊 Useful Commands

### View Logs
```powershell
# Real-time logs
heroku logs --tail

# Last 100 lines
heroku logs -n 100
```

### Check App Status
```powershell
heroku ps
# Should show: web.1: up
```

### Restart App
```powershell
heroku restart
```

### Open App
```powershell
heroku open
```

### Database Console
```powershell
heroku pg:psql
```

### View Configuration
```powershell
heroku config
```

---

## 🐛 Troubleshooting

### Issue: "heroku: command not found"

**Solution:** Install Heroku CLI and restart PowerShell

```powershell
# Download from: https://devcenter.heroku.com/articles/heroku-cli
# Then restart PowerShell
```

---

### Issue: "npm: command not found"

**Solution:** Install Node.js

```powershell
# Download from: https://nodejs.org/
# Install and restart PowerShell
```

---

### Issue: App crashes on startup

**Check logs:**
```powershell
heroku logs --tail
```

**Common fixes:**
```powershell
# 1. Ensure database is initialized
heroku run "python -c `"from backend.app import app, db; app.app_context().push(); db.create_all()`""

# 2. Check environment variables
heroku config

# 3. Restart
heroku restart
```

---

### Issue: "git push heroku main" fails

**If branch is named 'master':**
```powershell
git push heroku master
```

**If remote doesn't exist:**
```powershell
# Get your app name
heroku apps:info

# Add remote
heroku git:remote -a your-app-name
```

---

### Issue: Frontend doesn't load

**Ensure frontend was built:**
```powershell
cd frontend
npm run build
cd ..
git add frontend/build
git commit -m "Add built frontend"
git push heroku main
```

---

### Issue: Import errors from src/

This is already handled in `backend/app.py`. If you see errors, check that all files are committed:

```powershell
git status
git add .
git commit -m "Add missing files"
git push heroku main
```

---

## 🔄 Update Your Deployed App

When you make changes:

```powershell
# 1. Make changes to code

# 2. If frontend changed, rebuild
cd frontend
npm run build
cd ..

# 3. Commit changes
git add .
git commit -m "Your changes description"

# 4. Deploy
git push heroku main

# 5. App updates automatically!
```

---

## 📱 Where to Find Your App

### Get Your App URL

```powershell
# Method 1: Open in browser
heroku open

# Method 2: Get URL
heroku apps:info --json | ConvertFrom-Json | Select-Object -ExpandProperty app | Select-Object -ExpandProperty web_url

# Method 3: From dashboard
# Visit: https://dashboard.heroku.com/apps
```

Your URL will be something like:
- `https://signal-analyzer-12345.herokuapp.com`
- `https://your-chosen-name.herokuapp.com`

---

## 🎯 Quick Reference

### First Time Setup
```powershell
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create

# 4. Add database
heroku addons:create heroku-postgresql:mini

# 5. Set config
$SECRET_KEY = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
heroku config:set SECRET_KEY=$SECRET_KEY FLASK_ENV=production

# 6. Build frontend
cd frontend; npm install; npm run build; cd ..

# 7. Deploy
git add .; git commit -m "Deploy"; git push heroku main

# 8. Initialize DB
heroku run "python -c `"from backend.app import app, db; app.app_context().push(); db.create_all()`""

# 9. Open app
heroku open
```

---

## 💡 Alternative: Use Git Bash

If you have **Git for Windows** installed, you can use the bash script:

1. Open **Git Bash** (comes with Git for Windows)
2. Navigate to your project:
   ```bash
   cd /c/Users/csaba/DataChaEnhanced
   ```
3. Run the script:
   ```bash
   ./deploy_to_heroku.sh
   ```

---

## 📞 Need Help?

1. **Check logs:** `heroku logs --tail`
2. **View app info:** `heroku apps:info`
3. **Restart app:** `heroku restart`
4. **Check database:** `heroku pg:info`

---

## ✅ Deployment Checklist

- [ ] Heroku CLI installed
- [ ] Heroku account created
- [ ] Logged in (`heroku login`)
- [ ] App created (`heroku create`)
- [ ] PostgreSQL added
- [ ] Environment variables set
- [ ] Frontend built (`npm run build`)
- [ ] Code committed to git
- [ ] Pushed to Heroku
- [ ] Database initialized
- [ ] App opens successfully
- [ ] Tested file upload
- [ ] Tested analysis
- [ ] Tested export

---

## 🎉 Success!

Once deployed, your app will be accessible at:

**https://your-app-name.herokuapp.com**

No installation needed for users - just share the URL! 🚀

---

**Next:** Share this URL with colleagues and they can use the app immediately!

