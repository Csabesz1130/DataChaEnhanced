# Backend API Endpoint Review

## Summary
All core endpoints exist. Some advanced features need additional endpoints.

## Existing Endpoints

### Files API (`/api/files`)
- ✅ `POST /upload` - Upload ATF file
- ✅ `GET /<file_id>` - Get file info
- ✅ `GET /<file_id>/data` - Get parsed file data
- ✅ `GET /<file_id>/download` - Download original file
- ✅ `DELETE /<file_id>` - Delete file
- ✅ `GET /list` - List all files

### Analysis API (`/api/analysis`)
- ✅ `POST /process` - Process signal with action potential analysis
- ✅ `POST /integrals` - Calculate integrals for hyperpol/depol
- ✅ `GET /results/<analysis_id>` - Get analysis results
- ✅ `GET /list` - List all analyses
- ✅ `DELETE /<analysis_id>` - Delete analysis

### Filtering API (`/api/filter`)
- ✅ `POST /savgol` - Apply Savitzky-Golay filter
- ✅ `POST /butterworth` - Apply Butterworth filter
- ✅ `POST /wavelet` - Apply Wavelet filter
- ✅ `POST /combined` - Apply combined filters

### Export API (`/api/export`)
- ✅ `POST /excel` - Generate Excel export
- ✅ `POST /csv` - Generate CSV export
- ✅ `GET /download/<export_id>` - Download export file

### Health Check
- ✅ `GET /api/health` - Health check endpoint

## Missing Endpoints for Advanced Features

### Curve Fitting
- ❌ `POST /api/analysis/curve-fitting/linear` - Linear fit
- ❌ `POST /api/analysis/curve-fitting/exponential` - Exponential fit
- ❌ `GET /api/analysis/<analysis_id>/fits` - Get fit results

### Linear Fit Subtraction
- ❌ `POST /api/analysis/subtract-linear-fit` - Subtract linear fit from curves

### Starting Point Simulation
- ❌ `POST /api/analysis/simulate-starting-point` - Run starting point simulation

### Spike Removal
- ❌ `POST /api/analysis/<analysis_id>/remove-spikes` - Remove periodic spikes

### Batch Processing
- ❌ `POST /api/batch/create` - Create batch job
- ❌ `GET /api/batch/<job_id>` - Get batch job status
- ❌ `GET /api/batch/<job_id>/results` - Get batch results
- ❌ `POST /api/batch/<job_id>/cancel` - Cancel batch job

### History Management
- ✅ `GET /api/analysis/list` - Already exists, can be used for history
- ❌ `POST /api/analysis/<analysis_id>/restore` - Restore analysis from history
- ❌ `GET /api/analysis/history` - Get analysis history with filters

### Settings/Preferences
- ❌ `GET /api/settings` - Get user settings
- ❌ `POST /api/settings` - Save user settings

## API Integration Status

### Frontend Integration Status

#### Fully Integrated
- ✅ File upload (`/api/files/upload`)
- ✅ Analysis processing (`/api/analysis/process`)
- ✅ Export Excel (`/api/export/excel`)
- ✅ Export CSV (`/api/export/csv`)

#### Partially Integrated
- ⚠️ Analysis results (`/api/analysis/results/<id>`) - Exists but not used for history
- ⚠️ File list (`/api/files/list`) - Exists but not used in frontend

#### Not Integrated
- ❌ All filtering endpoints
- ❌ Integrals calculation (`/api/analysis/integrals`)
- ❌ Analysis list (`/api/analysis/list`)
- ❌ File deletion (`/api/files/<id>` DELETE)
- ❌ Analysis deletion (`/api/analysis/<id>` DELETE)

## Recommendations

### High Priority
1. Integrate filtering endpoints in frontend
2. Integrate integrals calculation endpoint
3. Add curve fitting endpoints to backend
4. Add starting point simulation endpoint

### Medium Priority
5. Add batch processing endpoints
6. Add spike removal endpoint
7. Add linear fit subtraction endpoint
8. Enhance history management endpoints

### Low Priority
9. Add settings/preferences endpoints
10. Add advanced export options

## Default Parameters

From `backend/config.py`:
```python
DEFAULT_ANALYSIS_PARAMS = {
    'n_cycles': 2,
    't0': 20,
    't1': 100,
    't2': 100,
    't3': 1000,
    'V0': -80,
    'V1': -100,
    'V2': -20,
    'cell_area_cm2': 1e-4
}
```

**Missing from defaults:**
- `n` (starting point) - default should be 35
- `auto_optimize_starting_point` - default True
- `use_alternative_method` - default False
- Integration ranges (hyperpol/depol)
- Normalization points

