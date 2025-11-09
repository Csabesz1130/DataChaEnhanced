# Quick Test Guide

## Starting the Application

### Option 1: Manual Start (Recommended for Testing)

**Terminal 1 - Backend:**
```bash
# Navigate to project root
cd C:\Users\csaba\DataChaEnhanced

# Install Python dependencies (if not already installed)
pip install -r backend/requirements.txt

# Set environment variables (Windows PowerShell)
$env:FLASK_APP="backend/app.py"
$env:FLASK_ENV="development"
$env:DATABASE_URL="sqlite:///test.db"  # Use SQLite for quick testing

# Start Flask backend
python backend/app.py
```

Backend should start on: http://localhost:5000

**Terminal 2 - Frontend:**
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies (if not already installed)
npm install

# Start React development server
npm start
```

Frontend should start on: http://localhost:3000

### Option 2: Quick Start Scripts

See below for automated startup scripts.

## Testing Checklist

1. **File Upload**
   - [ ] Drag & drop an .atf file
   - [ ] Verify progress bar appears
   - [ ] Check success notification
   - [ ] Verify file info displays

2. **Analysis Controls**
   - [ ] Check all parameters are visible
   - [ ] Test starting point (n) input
   - [ ] Toggle auto-optimize checkbox
   - [ ] Switch integration method
   - [ ] Verify validation works (try invalid values)

3. **Run Analysis**
   - [ ] Click "Run Analysis"
   - [ ] Verify loading state
   - [ ] Check success notification
   - [ ] Verify plot appears

4. **Plot Controls**
   - [ ] Toggle curve visibility
   - [ ] Test zoom controls
   - [ ] Toggle grid
   - [ ] Set custom axis limits
   - [ ] Export plot (PNG/SVG/PDF)

5. **Action Potential Tab**
   - [ ] Set integration ranges
   - [ ] Calculate integrals
   - [ ] Verify capacitance calculation
   - [ ] Check results display

6. **Filtering** (if data loaded)
   - [ ] Enable Savitzky-Golay filter
   - [ ] Apply filter
   - [ ] Check filter metrics
   - [ ] Test combined filters

7. **Export**
   - [ ] Open export options
   - [ ] Customize export settings
   - [ ] Export to Excel
   - [ ] Export to CSV

8. **Error Handling**
   - [ ] Try uploading invalid file
   - [ ] Try analysis with invalid parameters
   - [ ] Check error messages are user-friendly

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.8+)
- Install dependencies: `pip install -r backend/requirements.txt`
- Check port 5000 is not in use

### Frontend won't start
- Check Node.js version: `node --version` (need 14+)
- Install dependencies: `cd frontend && npm install`
- Check port 3000 is not in use
- Clear cache: `npm cache clean --force`

### API Connection Issues
- Verify backend is running on port 5000
- Check browser console for CORS errors
- Verify `REACT_APP_API_URL` in frontend/.env (if set)

### Database Issues
- For quick testing, SQLite is fine
- For production, use PostgreSQL
- Database tables auto-create on first run

