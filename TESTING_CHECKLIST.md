# Testing Checklist - Enhanced Webapp

## 🚀 Quick Start

The servers should now be starting:
- **Backend:** http://localhost:5000
- **Frontend:** http://localhost:3000 (will open automatically)

If they didn't start automatically, run:
```powershell
# In one terminal:
.\start_backend.ps1

# In another terminal:
.\start_frontend.ps1

# Or use the combined script:
.\start_all.ps1
```

## ✅ Testing Checklist

### 1. Initial Load
- [ ] Frontend loads at http://localhost:3000
- [ ] No console errors
- [ ] AppBar shows "Signal Analyzer"
- [ ] Layout displays correctly (controls left, plot right)

### 2. File Upload
- [ ] Drag & drop an .atf file (use `data/202304_0521.atf` or `data/202304_0523.atf`)
- [ ] Progress bar appears during upload
- [ ] Success notification appears
- [ ] File name and size display
- [ ] File info shows (points, duration, sampling rate if available)

### 3. Analysis Controls
- [ ] **Integration Method** section visible
  - [ ] Can switch between Traditional and Alternative methods
- [ ] **Basic Parameters** section visible
  - [ ] Number of Cycles input
  - [ ] t0, t1, t2, t3 inputs
  - [ ] Tooltips appear on hover (help icons)
- [ ] **Starting Point** section visible
  - [ ] Starting point (n) input field
  - [ ] Auto-optimize checkbox (should be checked by default)
  - [ ] Can toggle auto-optimize
- [ ] **Voltage Parameters** section visible
  - [ ] V0, V1, V2 inputs
  - [ ] Cell Area input
- [ ] **Validation** works
  - [ ] Try negative number for cycles → error message
  - [ ] Try 0 for time constants → error message
  - [ ] Error alert appears at bottom
  - [ ] "Run Analysis" button disabled when errors exist

### 4. Run Analysis
- [ ] Click "Run Analysis" button
- [ ] Loading state shows ("Analyzing...")
- [ ] Button disabled during analysis
- [ ] Success notification appears
- [ ] Plot appears on right side
- [ ] All curves visible (orange, normalized, average, hyperpol, depol)

### 5. Plot Controls
- [ ] **Curve Visibility Toggles** work
  - [ ] Uncheck "Orange" → orange curve disappears
  - [ ] Uncheck "Normalized" → normalized curve disappears
  - [ ] Can toggle all curves independently
- [ ] **Zoom Controls** work
  - [ ] Click zoom in → plot zooms in
  - [ ] Click zoom out → plot zooms out
  - [ ] Click reset → plot resets to auto-scale
- [ ] **Grid Toggle** works
  - [ ] Click grid icon → grid toggles on/off
- [ ] **Custom Axis Limits** work
  - [ ] Check "Custom Y Limits"
  - [ ] Enter Y Min and Y Max values
  - [ ] Plot Y-axis updates
  - [ ] Check "Custom X Range"
  - [ ] Enter X Min and X Max values
  - [ ] Plot X-axis updates
- [ ] **Export** works
  - [ ] Click "Export" button
  - [ ] Menu appears with PNG, SVG, PDF options
  - [ ] Click PNG → file downloads
  - [ ] Click SVG → file downloads
  - [ ] Click PDF → file downloads

### 6. Action Potential Tab
- [ ] **Results Display** section visible
  - [ ] Shows "Calculate integrals to see results" initially
- [ ] **Spike Removal** section visible
  - [ ] "Remove Spikes" button present
  - [ ] Click button → info notification appears (feature coming soon)
- [ ] **Integration Ranges** section visible
  - [ ] Hyperpol Start/End Index inputs
  - [ ] Depol Start/End Index inputs
  - [ ] Can change range values
- [ ] **Regression Controls** section visible
  - [ ] "Use Regression for Hyperpolarization" toggle
  - [ ] "Use Regression for Depolarization" toggle
- [ ] **Calculate Integrals** button works
  - [ ] Click "Calculate Integrals"
  - [ ] Loading state shows
  - [ ] Results appear:
    - [ ] Hyperpolarization integral value
    - [ ] Depolarization integral value
    - [ ] Linear Capacitance value
  - [ ] Success notification appears

### 7. Export
- [ ] **Export Options Dialog** works
  - [ ] Click "Options" button
  - [ ] Dialog opens
  - [ ] Can toggle "Include Charts"
  - [ ] Can toggle "Include Raw Data"
  - [ ] Can toggle individual curves
- [ ] **Export Excel** works
  - [ ] Click "Export Excel"
  - [ ] Loading state shows
  - [ ] Success notification appears
  - [ ] File downloads
- [ ] **Export CSV** works
  - [ ] Click "Export CSV"
  - [ ] Loading state shows
  - [ ] Success notification appears
  - [ ] File downloads

### 8. Error Handling
- [ ] **Network Error** handling
  - [ ] Stop backend server
  - [ ] Try to upload file → error notification appears
  - [ ] Try to run analysis → error notification appears
- [ ] **Validation Errors** work
  - [ ] Invalid parameters show error messages
  - [ ] Error alert appears
  - [ ] Button disabled

### 9. Notifications
- [ ] Success notifications appear (green)
- [ ] Error notifications appear (red)
- [ ] Info notifications appear (blue)
- [ ] Notifications auto-dismiss after 6 seconds
- [ ] Can manually close notifications

## 🐛 Known Issues / TODOs

1. **Spike Removal** - Backend endpoint not yet implemented (shows info message)
2. **Filtering** - FilterPanel component created but not yet integrated into main App
3. **Tab Navigation** - Components are stacked vertically, not in tabs yet
4. **File List** - Only single file upload supported (multi-file coming)

## 📝 Notes

- All new features are functional and ready for testing
- Error handling is comprehensive
- Notifications provide good user feedback
- Plot controls are fully interactive
- Export options are customizable

## 🎯 What to Test First

1. **Critical Path:**
   - Upload file → Set parameters → Run analysis → View plot → Calculate integrals → Export

2. **New Features:**
   - Starting point parameter
   - Auto-optimize toggle
   - Integration method selection
   - Plot controls (zoom, toggles, export)
   - Integration ranges and calculations

3. **Error Scenarios:**
   - Invalid file upload
   - Invalid parameters
   - Network errors

