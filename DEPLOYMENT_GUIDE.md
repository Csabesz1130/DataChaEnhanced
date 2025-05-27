# Signal Analyzer - Deployment Guide

This guide will help you set up automatic building and distribution of your Signal Analyzer application.

## 🎯 Overview

You have several options for distributing your application:

1. **GitHub Releases** (Recommended - Free & Automatic)
2. **Self-hosted Web Server** 
3. **Cloud Storage** (Google Drive, OneDrive, etc.)
4. **Multiple Platforms** (Using the Release Manager)

## 📋 Prerequisites

```bash
# Install required packages
pip install pyinstaller pillow requests flask google-api-python-client google-auth
```

## 🚀 Quick Start (GitHub Releases)

### Step 1: Replace the build.py file
Replace your current `build.py` with the enhanced version I provided. This new version:
- ✅ Creates proper EXE files with all dependencies
- ✅ Generates user-friendly ZIP packages
- ✅ Includes version information
- ✅ Tests the executable automatically

### Step 2: Set up GitHub Actions
1. Create the directory structure in your project:
   ```
   .github/
   └── workflows/
       └── build-release.yml
   ```

2. Copy the GitHub Actions workflow file I provided into `.github/workflows/build-release.yml`

### Step 3: Create a GitHub Release
```bash
# Method 1: Using Git tags (triggers automatic build)
git tag v1.0.0
git push origin v1.0.0

# Method 2: Using GitHub's web interface
# Go to your repo → Releases → Create a new release
```

### Step 4: Share the Download Link
Your teachers can always download the latest version from:
```
https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest
```

## 🛠️ Method 1: GitHub Releases (Recommended)

### Advantages:
- ✅ **Free** 
- ✅ **Automatic building** when you push code
- ✅ **Version management**
- ✅ **Always accessible** download link
- ✅ **No server maintenance**

### Setup:
1. **Push your code to GitHub**
2. **Add the GitHub Actions workflow** (provided above)
3. **Create releases using git tags:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

### For Teachers:
- **Always-current download link:** `https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest`
- **Specific version downloads:** Available on the releases page
- **No registration required** for downloading

## 🌐 Method 2: Self-Hosted Web Server

### Advantages:
- ✅ **Complete control**
- ✅ **Custom branding**
- ✅ **Download statistics**
- ✅ **Auto-updating download page**

### Setup:

1. **Install the web server:**
   ```bash
   pip install flask
   ```

2. **Run the distribution server:**
   ```bash
   python distribution_server.py --host 0.0.0.0 --port 5000
   ```

3. **Build and upload releases:**
   ```bash
   # Build your application
   python build.py
   
   # Upload to your server (copy ZIP file to 'releases' folder)
   cp dist/SignalAnalyzer_*.zip releases/
   ```

4. **Share the server URL:**
   ```
   http://YOUR_SERVER_IP:5000
   ```

### For Cloud Deployment:
You can deploy this on services like:
- **Heroku** (Free tier available)
- **Railway** (Simple deployment)
- **DigitalOcean** ($5/month)
- **AWS EC2** (Free tier available)

## 🔄 Method 3: Universal Release Manager

### Advantages:
- ✅ **Multiple platforms** (GitHub + Google Drive + FTP + more)
- ✅ **Automated notifications**
- ✅ **Configurable**

### Setup:

1. **Create configuration file:**
   ```bash
   python release_manager.py  # Creates release_config.ini
   ```

2. **Edit release_config.ini:**
   ```ini
   [DEFAULT]
   app_name = SignalAnalyzer
   version = 1.0.0
   description = Signal Analyzer - Advanced Signal Processing Tool

   [github]
   enabled = true
   repo = your-username/signal-analyzer
   token = your-github-token-here

   [google_drive]
   enabled = true
   folder_id = your-google-drive-folder-id
   credentials_file = credentials.json
   ```

3. **Run releases:**
   ```bash
   # Build and release to all configured platforms
   python release_manager.py --version 1.0.0
   
   # Build only (no upload)
   python release_manager.py --build-only
   ```

## 🔧 Manual Build Process

If you just want to create an EXE manually:

```bash
# Use the enhanced build script
python build.py

# Or use PyInstaller directly
pyinstaller --onedir --windowed --name=SignalAnalyzer run.py
```

The build script will create:
- `dist/SignalAnalyzer/` - Folder with executable and dependencies
- `dist/SignalAnalyzer_v1.0.0_TIMESTAMP.zip` - Distribution package

## 📱 Easy Update Workflow

### For You (Developer):
```bash
# Make your changes to the code
git add .
git commit -m "Updated signal processing algorithm"

# Create a new release
git tag v1.0.1
git push origin v1.0.1

# GitHub automatically builds and publishes the new version
```

### For Teachers:
- **Same download link always works**
- **No need to check for updates manually**
- **Students always get the latest version**

## 🎓 Recommended Setup for Educational Use

### Best Approach:
1. **Use GitHub Releases** for automatic building
2. **Create a simple landing page** with the download link
3. **Set up a simple redirect** if you want a custom URL

### Landing Page Example:
Create a simple HTML file:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Signal Analyzer Download</title>
    <meta http-equiv="refresh" content="0; url=https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest">
</head>
<body>
    <p>Redirecting to download page...</p>
    <p>If not redirected, <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest">click here</a></p>
</body>
</html>
```

Host this on GitHub Pages, Netlify, or any web hosting service.

## 🔗 Custom Domain Setup

If you want a custom URL like `signalanalyzer.yourschool.edu`:

1. **Host the landing page** on your school's web server
2. **Set up a redirect** to your GitHub releases page
3. **Use a URL shortener** like bit.ly or your school's URL shortener

## 📊 Monitoring and Analytics

### GitHub Releases:
- **Download counts** are automatically tracked
- **Release notes** can include changelogs
- **Issues/feedback** can be collected via GitHub Issues

### Self-hosted Server:
- **Real-time download statistics**
- **User analytics** (if needed)
- **Custom feedback forms**

## 🆘 Troubleshooting

### Build Issues:
```bash
# If build fails, try:
pip install --upgrade pyinstaller
pip install --upgrade pillow

# Check dependencies:
python test_imports.py

# Clean build:
rm -rf build dist *.spec
python build.py
```

### GitHub Actions Issues:
- Check the **Actions** tab in your GitHub repo
- Ensure **requirements.txt** includes all dependencies
- Verify the **run.py** file exists in your project root

### Distribution Issues:
- Test the EXE on a **clean Windows machine**
- Ensure all **DLL dependencies** are included
- Check **antivirus software** isn't blocking the executable

## 🎉 Summary

**For the simplest setup that works reliably:**

1. **Replace build.py** with the enhanced version
2. **Add GitHub Actions workflow**
3. **Push your code to GitHub**
4. **Create releases with git tags**
5. **Share the GitHub releases URL with teachers**

This gives you:
- ✅ Automatic building when you update code
- ✅ Always-accessible download link
- ✅ Version management
- ✅ No server costs or maintenance
- ✅ Professional appearance

The download link `https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest` will always redirect to the newest version, so teachers can bookmark it and students will always get the latest version.