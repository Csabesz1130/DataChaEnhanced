# Desktop Application Feature Inventory

## Main Tabs

### 1. Filters Tab (`src/gui/filter_tab.py`)
**Purpose:** Signal filtering controls

**Features:**
- Savitzky-Golay filter
  - Enable/disable toggle
  - Window length control (5-101)
  - Polynomial order control (2-5)
- Butterworth filter
  - Enable/disable toggle
  - Cutoff frequency control
  - Filter order control
- Wavelet filter
  - Enable/disable toggle
  - Wavelet level control
- Extract-Add filter
  - Enable/disable toggle
  - Prominence control
  - Width range controls (min/max)

**Status in Webapp:** ❌ Missing

### 2. Analysis Tab (`src/gui/analysis_tab.py`)
**Purpose:** Signal statistics and peak/event analysis

**Features:**
- Signal Statistics Display
  - Basic statistics (mean, std, min, max)
  - Refresh button
- Peak Detection
  - Enable/disable toggle
  - Prominence control
  - Distance control
  - Width control
  - Height control
  - Peak statistics display
- Event Analysis
  - Enable/disable toggle
  - Threshold control
  - Minimum event duration
  - Event statistics display

**Status in Webapp:** ❌ Missing

### 3. View Tab (`src/gui/view_tab.py`)
**Purpose:** Plot display controls

**Features:**
- Display Options
  - Show original signal toggle
  - Show filtered signal toggle
  - Show grid toggle
- Time Interval Selection
  - Enable custom interval toggle
  - Start time control
  - End time control
- Axis Limits
  - Custom Y limits toggle
  - Y min/max controls

**Status in Webapp:** ❌ Missing (partially covered by PlotViewer)

### 4. Action Potential Tab (`src/gui/action_potential_tab.py`)
**Purpose:** Main analysis controls and results

**Features:**
- **Results Display**
  - Integral value display
  - Purple curves integrals (hyperpol/depol)
  - Linear capacitance display
  - Progress bar
  - Status text

- **Spike Removal**
  - Remove Spikes button
  - Removes periodic spikes at (n + 200*i)

- **Parameters**
  - Integration method selection (Traditional vs Averaged Normalized)
  - Number of cycles
  - Time constants (t0, t1, t2)
  - Voltage levels (V0, V1, V2)
  - Cell area

- **Normalization Point**
  - Starting point (n) input
  - "Find Optimal Point" button (starting point simulation)
  - Auto-optimize starting point checkbox

- **Integration Range Controls**
  - Range selection manager
  - Hyperpol range (start/end)
  - Depol range (start/end)
  - Visual range selection on plot

- **Regression Controls**
  - Enable Points & Regression toggle
  - Use regression for hyperpol toggle
  - Use regression for depol toggle
  - Range selection on purple curves

- **Curve Fitting Panel** (integrated)
  - Linear fit for hyperpol/depol
  - Exponential fit for hyperpol/depol
  - Fit results display
  - Clear all fits
  - Export results

**Status in Webapp:** ❌ Missing (partially covered by AnalysisControls)

### 5. AI Analysis Tab (`src/gui/ai_analysis_tab.py`)
**Purpose:** AI-enhanced analysis features

**Status in Webapp:** ❌ Missing (optional feature)

### 6. Excel Learning Tab (`src/gui/excel_learning_tab.py`)
**Purpose:** Excel file learning background tasks

**Features:**
- Submit Task Tab
- Monitor Tasks Tab
- Results Tab
- Status bar

**Status in Webapp:** ❌ Missing (optional feature)

## Advanced Features

### Curve Fitting (`src/gui/curve_fitting_gui.py`)
**Features:**
- Manual curve fitting panel
- Linear fit (select 2 points)
- Exponential fit (select 1 start point)
- Separate controls for hyperpol and depol
- Fit results display (parameters, R²)
- Clear all fits
- Export results

**Status in Webapp:** ❌ Missing

### Linear Fit Subtraction (`src/analysis/linear_fit_subtractor.py`)
**Features:**
- Subtract linear fit from hyperpol curve
- Subtract linear fit from depol curve
- Subtract both curves
- Reset functionality
- Plot reload after subtraction

**Status in Webapp:** ❌ Missing

### Starting Point Simulation (`src/analysis/starting_point_simulator.py`)
**Features:**
- Test range of starting points (default 10-100)
- Quality metrics (smoothness score, outstanding points)
- Recommended starting point
- One-click application
- Visualization of results

**Status in Webapp:** ❌ Missing

### Spike Removal (`src/gui/direct_spike_removal.py`)
**Features:**
- Remove periodic spikes at (n + 200*i)
- Applied to all curves
- Integrated in Action Potential Tab

**Status in Webapp:** ❌ Missing

### Batch Processing
**Features:**
- Multi-file analysis (`src/gui/multi_file_analysis.py`)
- Batch set export (`src/gui/batch_set_exporter.py`)
- Simplified set export (`src/gui/simplified_set_exporter.py`)

**Status in Webapp:** ❌ Missing

## Export Features

### Excel Export
- Basic Excel export (`src/excel_export/excel_export.py`)
- Enhanced Excel export with charts (`src/excel_charted/dual_curves_export_integration.py`)
- Purple curves export
- Set-based export

**Status in Webapp:** ⚠️ Partial (basic Excel export exists)

### CSV Export
- Dual curves CSV export (`src/csv_export/dual_curves_csv_export.py`)

**Status in Webapp:** ⚠️ Partial (basic CSV export exists)

## UI Features

### History Management (`src/utils/analysis_history_manager.py`)
**Features:**
- Analysis history tracking
- History window (`src/gui/history_window.py`)
- View previous analyses
- Restore from history

**Status in Webapp:** ❌ Missing

### Window Management (`src/gui/window_manager.py`)
**Features:**
- Multiple plot windows
- Window tiling
- Window management

**Status in Webapp:** ❌ Missing (not needed for web)

### Plot Features
**Features:**
- Saved plot limits (xlim, ylim)
- Auto-restore plot limits
- Display mode toggles
- Multiple curve types
- Interactive zoom/pan

**Status in Webapp:** ⚠️ Partial (basic plot exists, missing advanced features)

## Menu Bar Features

### File Menu
- Load Data
- Export Data
- Export Figure
- Exit

### Analysis Menu
- View History
- Export Purple Curves

### Help Menu
- About
- Check for Updates

**Status in Webapp:** ❌ Missing (not applicable to web)

## Toolbar Features

### Excel Export
- Export to Excel with charts
- Dual curves export

### CSV Export
- Export to CSV
- Dual curves CSV export

### Set Export
- Simplified set export
- Batch set export

**Status in Webapp:** ⚠️ Partial (basic export exists)

## Summary

### Fully Implemented in Webapp
- ✅ Basic file upload
- ✅ Basic analysis processing
- ✅ Basic plot display
- ✅ Basic Excel/CSV export

### Partially Implemented
- ⚠️ Analysis controls (missing many parameters)
- ⚠️ Plot viewer (missing advanced controls)
- ⚠️ Export (missing options and customization)

### Missing
- ❌ Filtering interface
- ❌ Analysis tab (statistics, peaks, events)
- ❌ View tab (display controls)
- ❌ Action Potential tab (advanced controls)
- ❌ Curve fitting
- ❌ Linear fit subtraction
- ❌ Starting point simulation
- ❌ Spike removal
- ❌ Batch processing
- ❌ History management
- ❌ Advanced export options

