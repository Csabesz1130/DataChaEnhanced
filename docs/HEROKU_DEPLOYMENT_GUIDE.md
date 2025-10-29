# 🚀 Heroku Deployment Guide - Signal Analyzer Web App

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Heroku Account**: Sign up at [heroku.com](https://signup.heroku.com)
2. **Heroku CLI**: Install from [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)
3. **Git**: Ensure your project is in a git repository
4. **Node.js** (v14+) and **npm**: For frontend build
5. **Python** (3.11+): Tested version

---

## 🎯 Quick Start (5 Minutes)

```bash
# 1. Login to Heroku
heroku login

# 2. Create Heroku app
heroku create signal-analyzer-app

# 3. Add PostgreSQL addon
heroku addons:create heroku-postgresql:mini

# 4. Set environment variables
heroku config:set SECRET_KEY=$(openssl rand -hex 32)
heroku config:set FLASK_ENV=production

# 5. Deploy
git push heroku main

# 6. Initialize database
heroku run python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"

# 7. Open your app!
heroku open
```

---

## 📦 Detailed Step-by-Step Guide

### Step 1: Prepare Your Project

#### 1.1 Ensure All Files Are Created

Check that you have:
```
✓ backend/app.py
✓ backend/config.py
✓ backend/routes/
✓ backend/utils/
✓ backend/requirements.txt
✓ Procfile
✓ runtime.txt
✓ frontend/ (React app)
```

#### 1.2 Test Locally First

**Backend:**
```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=backend/app.py
export FLASK_ENV=development
export DATABASE_URL=postgresql://localhost/signal_analyzer

# Run Flask app
flask run
```

**Frontend:**
```bash
# Install dependencies
cd frontend
npm install

# Start dev server
npm start
```

Visit `http://localhost:3000` to test locally.

---

### Step 2: Set Up Heroku App

#### 2.1 Login to Heroku
```bash
heroku login
```

#### 2.2 Create Heroku App
```bash
# Create app with custom name
heroku create signal-analyzer-app

# OR let Heroku generate a name
heroku create

# Your app URL will be: https://signal-analyzer-app.herokuapp.com
```

#### 2.3 Verify Git Remote
```bash
git remote -v

# Should see:
# heroku  https://git.heroku.com/signal-analyzer-app.git (fetch)
# heroku  https://git.heroku.com/signal-analyzer-app.git (push)
```

---

### Step 3: Add Heroku Addons

#### 3.1 PostgreSQL Database
```bash
# Add PostgreSQL (Free tier: 10K rows)
heroku addons:create heroku-postgresql:mini

# Verify
heroku config:get DATABASE_URL
# Should see: postgres://...
```

#### 3.2 Redis (Optional - for caching/queue)
```bash
# Add Redis (Free tier: 25MB)
heroku addons:create heroku-redis:mini

# Verify
heroku config:get REDIS_URL
```

---

### Step 4: Configure Environment Variables

```bash
# Required variables
heroku config:set SECRET_KEY=$(openssl rand -hex 32)
heroku config:set FLASK_ENV=production
heroku config:set MAX_UPLOAD_SIZE=52428800

# Frontend URL (your Heroku app URL)
heroku config:set FRONTEND_URL=https://signal-analyzer-app.herokuapp.com

# Verify all config
heroku config
```

---

### Step 5: Build Frontend

Before deploying, build the React frontend:

```bash
cd frontend
npm install
npm run build

# This creates frontend/build/ directory
# with production-optimized static files
```

**Option A: Serve Frontend from Flask (Simpler)**

Update `backend/app.py` to serve static files:

```python
import os
from flask import send_from_directory

# Add after app creation
# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    frontend_folder = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build')
    if path != "" and os.path.exists(os.path.join(frontend_folder, path)):
        return send_from_directory(frontend_folder, path)
    else:
        return send_from_directory(frontend_folder, 'index.html')
```

**Option B: Separate Frontend Deployment (Better for scaling)**

Deploy frontend to Netlify/Vercel:
1. Push frontend to separate repo
2. Connect to Netlify/Vercel
3. Set `REACT_APP_API_URL` to your Heroku backend URL

---

### Step 6: Deploy to Heroku

#### 6.1 Commit All Changes
```bash
git add .
git commit -m "Prepare for Heroku deployment"
```

#### 6.2 Push to Heroku
```bash
git push heroku main

# Watch the build process
# Should see:
# - Python buildpack installing
# - Dependencies from requirements.txt installing
# - Gunicorn starting
```

#### 6.3 Check Deployment Status
```bash
# View logs
heroku logs --tail

# Check dyno status
heroku ps

# Should see:
# web.1: up 2025/01/01 12:00:00 (~ 1m ago)
```

---

### Step 7: Initialize Database

```bash
# Run database initialization
heroku run python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"

# Verify tables were created
heroku pg:psql
# In psql:
\dt
# Should see: uploaded_files, analysis_results, export_files, sessions
\q
```

---

### Step 8: Test Your Deployment

#### 8.1 Open Your App
```bash
heroku open
```

#### 8.2 Test API Endpoints
```bash
# Health check
curl https://signal-analyzer-app.herokuapp.com/api/health

# Should return:
# {"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

#### 8.3 Test File Upload & Analysis

1. Open app in browser
2. Upload an .atf file
3. Set parameters
4. Run analysis
5. View plots
6. Export results

---

## 🔧 Troubleshooting

### Issue: App Crashes on Startup

**Check logs:**
```bash
heroku logs --tail
```

**Common fixes:**
```bash
# 1. Ensure DATABASE_URL is set
heroku config:get DATABASE_URL

# 2. Check Python version
cat runtime.txt
# Should be: python-3.11.5

# 3. Verify Procfile
cat Procfile
# Should be: web: gunicorn backend.app:app ...

# 4. Restart dynos
heroku restart
```

---

### Issue: Database Connection Fails

**Fix DATABASE_URL format:**
```bash
# Heroku sets it as postgres://
# But SQLAlchemy needs postgresql://
# Our config.py handles this, but verify:

heroku config:get DATABASE_URL
# If starts with postgres://, config.py converts it
```

**Recreate database:**
```bash
heroku pg:reset DATABASE
heroku run python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"
```

---

### Issue: File Uploads Fail

**Check upload limits:**
```bash
# Increase if needed
heroku config:set MAX_UPLOAD_SIZE=104857600  # 100 MB
```

**Use S3 for production:**
```bash
# Add AWS credentials
heroku config:set AWS_ACCESS_KEY_ID=your-key
heroku config:set AWS_SECRET_ACCESS_KEY=your-secret
heroku config:set S3_BUCKET=your-bucket
```

---

### Issue: Slow Performance

**Scale up dynos:**
```bash
# Check current dyno type
heroku ps

# Upgrade to Hobby dyno (never sleeps)
heroku ps:resize web=hobby

# Or Professional dyno (more power)
heroku ps:resize web=standard-1x
```

**Add more workers:**
```bash
# Scale to 2 web dynos
heroku ps:scale web=2
```

---

## 📊 Monitoring & Maintenance

### View Logs
```bash
# Real-time logs
heroku logs --tail

# Last 200 lines
heroku logs -n 200

# Filter for errors
heroku logs --tail | grep ERROR
```

### Database Management
```bash
# Connect to database
heroku pg:psql

# Database info
heroku pg:info

# Backup database
heroku pg:backups:capture
heroku pg:backups:download
```

### Performance Monitoring
```bash
# View metrics
heroku logs --tail --dyno web

# Or use Heroku dashboard:
# https://dashboard.heroku.com/apps/signal-analyzer-app/metrics
```

---

## 🔒 Security Best Practices

### 1. Rotate Secret Keys
```bash
heroku config:set SECRET_KEY=$(openssl rand -hex 32)
```

### 2. Use HTTPS Only
Add to `backend/app.py`:
```python
from flask_talisman import Talisman
Talisman(app, content_security_policy=None)
```

### 3. Set Up Rate Limiting
```python
# Add Flask-Limiter
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/files/upload')
@limiter.limit("10 per hour")
def upload():
    ...
```

### 4. Enable CORS Properly
Already configured in `backend/app.py` with specific origins.

---

## 💰 Cost Optimization

### Free Tier Resources
- **Dyno**: 550 free hours/month (sleeps after 30 min inactivity)
- **PostgreSQL**: 10,000 rows free
- **Redis**: 25 MB free

### Tips to Stay Free
1. Use eco dyno (free tier)
2. App sleeps after inactivity (wakes on request)
3. Clean up old files regularly
4. Limit database records

### Upgrade When Needed
```bash
# Hobby dyno: $7/month (never sleeps)
heroku ps:resize web=hobby

# Standard-1X: $25/month (better performance)
heroku ps:resize web=standard-1x

# PostgreSQL Mini: $9/month (10M rows)
heroku addons:upgrade heroku-postgresql:mini
```

---

## 🚀 CI/CD Setup (Optional)

### Automatic Deployments from GitHub

1. Connect GitHub to Heroku:
   ```bash
   # On Heroku Dashboard:
   # App → Deploy → GitHub → Connect Repository
   ```

2. Enable automatic deploys:
   - Choose branch (e.g., `main`)
   - Enable "Wait for CI to pass"
   - Click "Enable Automatic Deploys"

3. Create `.github/workflows/test.yml`:
   ```yaml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: actions/setup-python@v2
           with:
             python-version: 3.11
         - name: Install dependencies
           run: pip install -r backend/requirements.txt
         - name: Run tests
           run: pytest
   ```

---

## 📝 Deployment Checklist

Before going to production:

- [ ] Tested locally with PostgreSQL
- [ ] All environment variables set
- [ ] Secret key is random and secure
- [ ] Database tables initialized
- [ ] File uploads working
- [ ] Analysis functions working
- [ ] Export functions working
- [ ] Frontend loads correctly
- [ ] API endpoints responding
- [ ] CORS configured properly
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Backup strategy planned
- [ ] Monitoring set up
- [ ] Documentation updated

---

## 🎓 Next Steps

1. **Custom Domain**: 
   ```bash
   heroku domains:add www.your-domain.com
   ```

2. **SSL Certificate** (automatic with Heroku)

3. **Add Authentication**:
   - Implement Flask-JWT-Extended
   - Add user registration/login
   - Protect API endpoints

4. **Background Jobs**:
   - Add Celery worker
   - Process long-running analyses async

5. **Caching**:
   - Implement Redis caching
   - Cache analysis results
   - Speed up repeated requests

---

## 📞 Support & Resources

- **Heroku Docs**: https://devcenter.heroku.com
- **Flask Docs**: https://flask.palletsprojects.com
- **React Docs**: https://react.dev
- **PostgreSQL Docs**: https://www.postgresql.org/docs

---

**Congratulations! Your Signal Analyzer is now deployed on Heroku! 🎉**

