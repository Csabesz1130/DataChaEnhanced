# Frontend Component Gap Analysis

## Executive Summary
This document identifies gaps between the current webapp frontend and the desktop application features.

## 1. FileUpload Component Analysis

### Current Implementation
- ✅ Drag & drop support
- ✅ File upload to backend
- ✅ Basic error handling
- ✅ File type validation (.atf, .txt)
- ✅ Upload progress indicator
- ✅ Success state display

### Missing Features (vs Desktop)
- ❌ File list view (multiple files)
- ❌ File info display (size, points, duration, sampling rate from data_info)
- ❌ File deletion capability
- ❌ File selection/deselection
- ❌ File download capability
- ❌ File metadata display
- ❌ Recent files list
- ❌ File comparison view

## 2. AnalysisControls Component Analysis

### Current Implementation
- ✅ Basic parameters: n_cycles, t0, t1, t2, t3, V0, V1, V2, cell_area_cm2
- ✅ Accordion layout for organization
- ✅ Form submission handling
- ✅ Disabled state when no file

### Missing Features (vs Desktop ActionPotentialTab)
- ❌ **Starting point (n)** parameter - critical for analysis
- ❌ **Auto-optimize starting point** toggle
- ❌ **Use alternative method** toggle
- ❌ **Integration method** selection (direct vs alternative)
- ❌ **Normalization points** controls (hyperpol/depol normalization)
- ❌ **Integration ranges** controls (hyperpol/depol ranges)
- ❌ **Regression controls** (use regression for hyperpol/depol)
- ❌ **Display options** (show/hide various curves)
- ❌ Parameter validation (min/max values)
- ❌ Parameter presets/saved configurations
- ❌ Parameter import/export (JSON)
- ❌ Tooltips/help text for parameters
- ❌ Parameter history/undo-redo
- ❌ Real-time parameter validation feedback

### Desktop Parameters Not in Webapp
From `action_potential_tab.py`:
- `n` (starting point) - default 35
- `auto_optimize` (BooleanVar)
- `integration_method` (StringVar: "direct" or "alternative")
- `use_regression_hyperpol` (BooleanVar)
- `use_regression_depol` (BooleanVar)
- `normalization_point_hyperpol` (IntVar)
- `normalization_point_depol` (IntVar)
- `integration_start_hyperpol` (IntVar)
- `integration_end_hyperpol` (IntVar)
- `integration_start_depol` (IntVar)
- `integration_end_depol` (IntVar)
- Display mode toggles for various curves

## 3. PlotViewer Component Analysis

### Current Implementation
- ✅ Plotly.js integration
- ✅ Multiple curve types displayed (orange, normalized, average, modified hyperpol/depol)
- ✅ Basic legend
- ✅ Responsive layout
- ✅ Interactive zoom/pan (Plotly default)

### Missing Features (vs Desktop)
- ❌ **Curve visibility toggles** (show/hide individual curves)
- ❌ **Zoom controls** (custom buttons for zoom in/out/reset)
- ❌ **Axis controls** (custom Y limits, time range selection)
- ❌ **Cursor/measurement tool** (show values on hover - partially exists but not enhanced)
- ❌ **Legend with interactive toggles** (click to show/hide)
- ❌ **Plot export** (PNG, SVG, PDF)
- ❌ **Plot comparison mode** (overlay multiple analyses)
- ❌ **Grid toggle**
- ❌ **Axis labels customization**
- ❌ **Plot limits save/restore** (desktop has saved_plot_limits)
- ❌ **Multiple plot windows** (desktop has window_manager)
- ❌ **Plot refresh controls**

### Desktop Plot Features Not in Webapp
- Window manager for multiple plot windows
- Saved plot limits (xlim, ylim) with auto-restore
- Display mode toggles (show_noisy_original, show_red_curve, etc.)
- Processed/average/normalized/modified display modes
- Plot window management (new, close, tile)

## 4. ExportButton Component Analysis

### Current Implementation
- ✅ Excel export button
- ✅ CSV export button
- ✅ Loading state
- ✅ Success/error snackbar
- ✅ Download trigger

### Missing Features (vs Desktop)
- ❌ **Export options dialog** (include charts, raw data, etc.)
- ❌ **Export format selection** (currently hardcoded)
- ❌ **Export customization** (which curves to include)
- ❌ **Export preview**
- ❌ **Batch export** (multiple analyses)
- ❌ **Export history**
- ❌ **Purple curves export** (desktop has specific export for purple curves)
- ❌ **Set-based export** (desktop has set exporter)
- ❌ **Export with charts** (Excel charts integration)

### Desktop Export Features Not in Webapp
- Purple curves Excel export with charts
- Set-based export (multiple files as sets)
- Batch set export
- Export backup manager
- Enhanced Excel export with dual curves

## 5. Missing Major Components

### 5.1 Filtering Interface
**Status:** ❌ Completely Missing
- No filter panel component
- No Savitzky-Golay controls
- No Butterworth controls
- No Wavelet controls
- No combined filter interface
- No filter preview
- No filter metrics display

### 5.2 Action Potential Analysis Tab
**Status:** ❌ Completely Missing
- No spike removal interface
- No integration range controls
- No regression controls
- No normalization point controls
- No integral calculation display
- No capacitance calculation
- No results summary panel

### 5.3 Curve Fitting Interface
**Status:** ❌ Completely Missing
- No linear fit interface
- No exponential fit interface
- No fit results display
- No fit visualization
- No fit export

### 5.4 Linear Fit Subtraction
**Status:** ❌ Completely Missing
- No subtraction controls
- No subtract/reset buttons
- No status indicators

### 5.5 Starting Point Simulation
**Status:** ❌ Completely Missing
- No simulation interface
- No progress tracking
- No results display

### 5.6 Batch Processing
**Status:** ❌ Completely Missing
- No batch job interface
- No job queue management
- No progress tracking
- No results aggregation

### 5.7 History Management
**Status:** ❌ Completely Missing
- No analysis history list
- No history filtering/search
- No restore from history
- No history export

### 5.8 Settings & Preferences
**Status:** ❌ Completely Missing
- No theme selection
- No parameter presets management
- No plot preferences
- No keyboard shortcuts

## 6. Navigation & Layout Gaps

### Current Implementation
- ✅ Basic AppBar with title
- ✅ Two-column layout (controls left, plot right)
- ✅ Container with max width

### Missing Features
- ❌ Tab-based navigation (desktop has Filter, Analysis, View, Action Potential tabs)
- ❌ Collapsible sidebar
- ❌ Responsive layout (mobile-friendly)
- ❌ Keyboard shortcuts
- ❌ Breadcrumb navigation
- ❌ Workspace management

## 7. Error Handling Gaps

### Current Implementation
- ✅ Basic error display in App.js
- ✅ API error handling in components
- ✅ Console error logging

### Missing Features
- ❌ Global error handler
- ❌ User-friendly error messages
- ❌ Error logging service
- ❌ Retry logic for failed requests
- ❌ Offline detection
- ❌ Error recovery suggestions

## 8. State Management Gaps

### Current Implementation
- ✅ Basic React state in App.js
- ✅ Component-level state

### Missing Features
- ❌ Centralized state management (Redux/Context)
- ❌ State persistence (localStorage)
- ❌ Undo/redo functionality
- ❌ State history
- ❌ Session management

## 9. API Integration Gaps

### Current Implementation
- ✅ Basic axios setup
- ✅ File upload endpoint
- ✅ Analysis endpoint
- ✅ Export endpoints

### Missing Endpoints Integration
- ❌ Filter endpoints (savgol, butterworth, wavelet, combined)
- ❌ Analysis results retrieval endpoint
- ❌ Integrals calculation endpoint
- ❌ Analysis list endpoint
- ❌ File list endpoint
- ❌ File deletion endpoint
- ❌ Analysis deletion endpoint

## 10. Priority Summary

### Critical (Blocks Core Functionality)
1. Missing starting point (n) parameter
2. Missing auto-optimize toggle
3. Missing integration ranges controls
4. Missing normalization points
5. Missing regression controls

### High Priority (Major Features)
6. Filtering interface (complete)
7. Action Potential Analysis Tab (complete)
8. Curve visibility toggles in plot
9. Export options dialog
10. Error handling improvements

### Medium Priority (Enhancements)
11. Curve fitting interface
12. Linear fit subtraction
13. Starting point simulation
14. History management
15. Batch processing

### Low Priority (Polish)
16. Settings panel
17. Keyboard shortcuts
18. Workspace management
19. Advanced plot features
20. Performance optimizations

