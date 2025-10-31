# Heroku Web Deployment Architecture

## 🎯 Overview

Transform Signal Analyzer from a desktop Tkinter application to a **web-based application** deployed on Heroku, while **reusing existing analysis code**.

### Architecture: Desktop → Web

```
BEFORE (Desktop):
┌─────────────────────────────┐
│   Tkinter GUI (Desktop)     │
├─────────────────────────────┤
│   Analysis Logic            │
│   (numpy, scipy, matplotlib)│
├─────────────────────────────┤
│   File I/O (local files)    │
└─────────────────────────────┘

AFTER (Web):
┌──────────────────────────────────────────┐
│  React Frontend (Browser)                │
│  - Upload files                          │
│  - Interactive plots (Plotly.js)         │
│  - Real-time analysis controls           │
└──────────────────────────────────────────┘
              ↕ HTTP/WebSocket
┌──────────────────────────────────────────┐
│  Flask/FastAPI Backend (Heroku)          │
│  ┌────────────────────────────────────┐  │
│  │ REST API Endpoints                 │  │
│  ├────────────────────────────────────┤  │
│  │ REUSED Desktop Code:               │  │
│  │ - src/analysis/*                   │  │
│  │ - src/filtering/*                  │  │
│  │ - src/io_utils/*                   │  │
│  │ - src/excel_export/*               │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
              ↕
┌──────────────────────────────────────────┐
│  PostgreSQL (Heroku Addon)               │
│  - Store analysis results                │
│  - User sessions                         │
└──────────────────────────────────────────┘
```

---

## 🏗️ Architecture Components

### 1. Backend (Flask API)

**Purpose:** Expose desktop analysis code as REST API

**Technology Stack:**
- **Flask** or **FastAPI** (Python web framework)
- **Flask-CORS** (enable frontend communication)
- **Gunicorn** (production WSGI server)
- **PostgreSQL** (database via Heroku addon)
- **Redis** (optional - for caching/queue)

**Reused Desktop Code:**
- ✅ `src/analysis/action_potential.py` → API endpoint
- ✅ `src/filtering/filtering.py` → Filter endpoints
- ✅ `src/io_utils/io_utils.py` → File parsing
- ✅ `src/excel_export/` → Export generation

**New Backend Code:**
- 🆕 `backend/app.py` - Flask application
- 🆕 `backend/routes/` - API endpoints
- 🆕 `backend/models/` - Database models
- 🆕 `backend/utils/` - Web-specific utilities

### 2. Frontend (React)

**Purpose:** Modern web UI replacing Tkinter

**Technology Stack:**
- **React** (UI library)
- **Plotly.js** or **Chart.js** (interactive plots)
- **Axios** (HTTP client)
- **Material-UI** or **Tailwind CSS** (styling)
- **React-Dropzone** (file uploads)

**Features:**
- File upload (drag & drop)
- Real-time plotting
- Filter controls
- Analysis parameter adjustment
- Export results (Excel, CSV)
- Session management

### 3. Heroku Deployment

**Services:**
- **Heroku Dyno** (web process)
- **PostgreSQL addon** (database)
- **Redis addon** (optional - for tasks)
- **Heroku S3** or **Cloudinary** (file storage)

---

## 📦 Project Structure

```
DataChaEnhanced/
├── backend/                    # Flask API (NEW)
│   ├── app.py                 # Main Flask app
│   ├── requirements.txt       # Python dependencies
│   ├── Procfile              # Heroku process
│   ├── runtime.txt           # Python version
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analysis.py       # Analysis endpoints
│   │   ├── filtering.py      # Filter endpoints
│   │   ├── files.py          # File upload/download
│   │   └── export.py         # Export endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── session.py        # User session
│   │   └── analysis.py       # Analysis results
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_handler.py   # Handle uploads
│   │   ├── plot_converter.py # Matplotlib → JSON
│   │   └── db.py             # Database utilities
│   └── config.py             # Configuration
│
├── frontend/                   # React app (NEW)
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       ├── components/
│       │   ├── FileUpload.js
│       │   ├── PlotViewer.js
│       │   ├── FilterControls.js
│       │   ├── AnalysisControls.js
│       │   └── ExportButton.js
│       ├── services/
│       │   └── api.js         # Backend API client
│       └── utils/
│           └── plotHelpers.js
│
├── src/                        # EXISTING desktop code (REUSED)
│   ├── analysis/              # ✅ Reused in backend
│   ├── filtering/             # ✅ Reused in backend
│   ├── io_utils/              # ✅ Reused in backend
│   ├── excel_export/          # ✅ Reused in backend
│   └── gui/                   # ❌ Not used (replaced by React)
│
├── docs/
│   └── heroku_web_deployment.md
│
├── .env                       # Environment variables
├── .gitignore
└── README.md
```

---

## 🔄 Code Reuse Strategy

### ✅ What We Reuse (95% of analysis code!)

```python
# DESKTOP CODE (existing)
from src.analysis.action_potential import ActionPotentialProcessor
from src.filtering.filtering import apply_savgol_filter, butter_lowpass_filter
from src.io_utils.io_utils import ATFFileReader

# WEB BACKEND (wraps desktop code)
# backend/routes/analysis.py
@app.route('/api/analyze', methods=['POST'])
def analyze_signal():
    # Get uploaded file
    file = request.files['file']
    params = request.json.get('params')
    
    # REUSE desktop code!
    reader = ATFFileReader(file)
    data, time_data = reader.load_atf()
    
    processor = ActionPotentialProcessor(data, time_data, params)
    processor.process_signal()
    
    # Convert results to JSON
    results = {
        'orange_curve': processor.orange_curve.tolist(),
        'normalized_curve': processor.normalized_curve.tolist(),
        'integrals': processor.calculate_cleaned_integrals()
    }
    
    return jsonify(results)
```

**Benefits:**
- ✅ No rewriting analysis algorithms
- ✅ Proven, tested code
- ✅ Maintain one codebase
- ✅ Fast development

### ❌ What We Replace

| Desktop | Web Alternative |
|---------|----------------|
| `tkinter` GUI | React components |
| `matplotlib` (Tkinter backend) | Plotly.js / Chart.js |
| Local file system | Heroku file uploads + S3 |
| Desktop windows | Web pages |

---

## 🚀 API Design

### Core Endpoints

#### 1. File Management
```
POST   /api/files/upload           Upload ATF file
GET    /api/files/:id              Get file info
DELETE /api/files/:id              Delete file
```

#### 2. Analysis
```
POST   /api/analysis/process       Run action potential analysis
POST   /api/analysis/integrals     Calculate integrals
GET    /api/analysis/results/:id   Get analysis results
```

#### 3. Filtering
```
POST   /api/filter/savgol          Apply Savitzky-Golay filter
POST   /api/filter/butterworth     Apply Butterworth filter
POST   /api/filter/wavelet         Apply wavelet filter
POST   /api/filter/combined        Apply combined filters
```

#### 4. Export
```
POST   /api/export/excel           Generate Excel export
POST   /api/export/csv             Generate CSV export
GET    /api/export/download/:id    Download export file
```

#### 5. Session Management
```
POST   /api/session/create         Create new session
GET    /api/session/:id            Get session data
DELETE /api/session/:id            Clear session
```

### Example API Flow

```javascript
// Frontend (React)
const analyzeFile = async (file, params) => {
  // 1. Upload file
  const formData = new FormData();
  formData.append('file', file);
  
  const uploadResponse = await axios.post('/api/files/upload', formData);
  const fileId = uploadResponse.data.id;
  
  // 2. Run analysis
  const analysisResponse = await axios.post('/api/analysis/process', {
    file_id: fileId,
    params: {
      n_cycles: 2,
      t0: 20,
      t1: 100,
      // ... other params
    }
  });
  
  // 3. Get results
  const results = analysisResponse.data;
  
  // 4. Plot using Plotly
  Plotly.newPlot('plot', [{
    x: results.orange_curve_times,
    y: results.orange_curve,
    name: 'Orange Curve',
    type: 'scatter'
  }]);
  
  return results;
};
```

---

## 💾 Data Flow

### File Upload & Processing

```
1. User uploads .atf file in browser
   ↓
2. React sends multipart/form-data to Flask
   ↓
3. Flask saves to temporary storage
   ↓
4. Backend parses using existing ATFFileReader
   ↓
5. Data stored in PostgreSQL
   ↓
6. Returns JSON response with data
   ↓
7. React receives data and renders plot
```

### Real-time Analysis

```
User adjusts parameters
   ↓
Frontend sends POST with new params
   ↓
Backend reprocesses using existing code
   ↓
Returns new results as JSON
   ↓
Frontend updates plot (no page reload!)
```

---

## 🔐 Security Considerations

### File Upload Security
```python
# backend/utils/file_handler.py
ALLOWED_EXTENSIONS = {'atf', 'txt', 'csv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_upload(file):
    if not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Use secure filename
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    
    # Save with unique ID
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
    file.save(file_path)
    
    return file_id, file_path
```

### API Authentication (Optional)
```python
# backend/utils/auth.py
from flask_jwt_extended import JWTManager, create_access_token

# Simple JWT authentication
@app.route('/api/login', methods=['POST'])
def login():
    # For now, simple session-based
    # Can add user accounts later
    access_token = create_access_token(identity=str(uuid.uuid4()))
    return jsonify(access_token=access_token)
```

---

## 🎨 Frontend Components

### 1. File Upload Component

```javascript
// frontend/src/components/FileUpload.js
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileUpload }) => {
  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    onFileUpload(file);
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.atf', '.txt']
    },
    maxFiles: 1
  });

  return (
    <div {...getRootProps()} className="dropzone">
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop ATF file here...</p>
      ) : (
        <p>Drag & drop ATF file here, or click to select</p>
      )}
    </div>
  );
};

export default FileUpload;
```

### 2. Plot Viewer Component

```javascript
// frontend/src/components/PlotViewer.js
import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';

const PlotViewer = ({ data }) => {
  const plotRef = useRef(null);

  useEffect(() => {
    if (!data) return;

    const traces = [
      {
        x: data.orange_curve_times,
        y: data.orange_curve,
        name: 'Orange Curve',
        type: 'scatter',
        mode: 'lines',
        line: { color: 'orange', width: 2 }
      },
      {
        x: data.normalized_curve_times,
        y: data.normalized_curve,
        name: 'Normalized Curve',
        type: 'scatter',
        mode: 'lines',
        line: { color: 'blue', width: 2 }
      }
    ];

    const layout = {
      title: 'Signal Analysis',
      xaxis: { title: 'Time (s)' },
      yaxis: { title: 'Current (pA)' },
      showlegend: true,
      hovermode: 'closest'
    };

    Plotly.newPlot(plotRef.current, traces, layout, {
      responsive: true,
      displayModeBar: true
    });
  }, [data]);

  return <div ref={plotRef} style={{ width: '100%', height: '600px' }} />;
};

export default PlotViewer;
```

### 3. Analysis Controls Component

```javascript
// frontend/src/components/AnalysisControls.js
import React, { useState } from 'react';

const AnalysisControls = ({ onAnalyze }) => {
  const [params, setParams] = useState({
    n_cycles: 2,
    t0: 20,
    t1: 100,
    t2: 100,
    t3: 1000,
    V0: -80,
    V1: -100,
    V2: -20,
    cell_area_cm2: 0.0001
  });

  const handleChange = (e) => {
    setParams({
      ...params,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onAnalyze(params);
  };

  return (
    <form onSubmit={handleSubmit} className="controls-form">
      <h3>Analysis Parameters</h3>
      
      <label>
        Number of Cycles:
        <input
          type="number"
          name="n_cycles"
          value={params.n_cycles}
          onChange={handleChange}
        />
      </label>
      
      <label>
        t0 (ms):
        <input
          type="number"
          name="t0"
          value={params.t0}
          onChange={handleChange}
        />
      </label>
      
      {/* ... more parameters ... */}
      
      <button type="submit">Analyze</button>
    </form>
  );
};

export default AnalysisControls;
```

---

## 📊 Database Schema

```sql
-- PostgreSQL schema
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE uploaded_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    file_size INTEGER,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    file_id UUID REFERENCES uploaded_files(id),
    params JSONB NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE export_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analysis_results(id),
    export_type VARCHAR(50),  -- 'excel', 'csv'
    file_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_sessions_created ON sessions(created_at);
CREATE INDEX idx_files_session ON uploaded_files(session_id);
CREATE INDEX idx_analysis_session ON analysis_results(session_id);
```

---

## 🔧 Configuration

### Environment Variables

```bash
# .env (local development)
FLASK_APP=backend/app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

DATABASE_URL=postgresql://localhost/signal_analyzer
REDIS_URL=redis://localhost:6379

UPLOAD_FOLDER=/tmp/uploads
MAX_UPLOAD_SIZE=52428800  # 50 MB

# AWS S3 (for production file storage)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=signal-analyzer-uploads

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000
```

### Heroku Config Vars

```bash
# Set via Heroku CLI or Dashboard
heroku config:set SECRET_KEY=$(openssl rand -hex 32)
heroku config:set FLASK_ENV=production
heroku config:set MAX_UPLOAD_SIZE=52428800

# Database URL set automatically by PostgreSQL addon
# heroku addons:create heroku-postgresql:mini
```

---

## 📈 Performance Optimization

### 1. Caching

```python
# backend/utils/cache.py
from flask_caching import Cache

cache = Cache(config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL')
})

@app.route('/api/analysis/results/<id>')
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_analysis_results(id):
    results = db.get_analysis(id)
    return jsonify(results)
```

### 2. Background Tasks

```python
# backend/utils/tasks.py
from celery import Celery

celery = Celery(app.name, broker=os.getenv('REDIS_URL'))

@celery.task
def process_signal_async(file_id, params):
    """Process signal in background"""
    # Long-running analysis
    processor = ActionPotentialProcessor(data, time_data, params)
    results = processor.process_signal()
    
    # Store results in database
    db.save_analysis_results(file_id, results)
    
    return results

# In route:
@app.route('/api/analysis/process', methods=['POST'])
def process_signal():
    task = process_signal_async.delay(file_id, params)
    return jsonify({'task_id': task.id, 'status': 'processing'})
```

### 3. Response Compression

```python
from flask_compress import Compress

Compress(app)
```

---

## 🚀 Deployment Steps (Summary)

See detailed guide in deployment scripts, but overview:

### 1. Prepare Backend
```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare Frontend
```bash
cd frontend
npm install
npm run build
```

### 3. Deploy to Heroku
```bash
heroku login
heroku create signal-analyzer-app
heroku addons:create heroku-postgresql:mini
heroku addons:create heroku-redis:mini

git push heroku main
heroku open
```

---

## 📝 Next Steps

1. ✅ Review this architecture
2. 🔨 Create backend API implementation
3. 🎨 Create frontend React app
4. 🐘 Set up PostgreSQL models
5. 🚀 Create Heroku deployment configs
6. 📖 Create step-by-step deployment guide

---

## 💰 Heroku Pricing Estimate

| Resource | Free Tier | Paid Option |
|----------|-----------|-------------|
| Web Dyno | $0 (550 hrs/month) | $7/month (Hobby) |
| PostgreSQL | $0 (10K rows) | $9/month (Mini) |
| Redis | $0 (25 MB) | $15/month (Mini) |
| **Total** | **$0** | **~$30/month** |

**Recommendation:** Start with free tier, upgrade as needed.

---

## 🎯 Success Criteria

- ✅ Upload .atf files via web interface
- ✅ Run analysis with custom parameters
- ✅ View interactive plots in browser
- ✅ Apply filters in real-time
- ✅ Export results to Excel/CSV
- ✅ No desktop installation required
- ✅ Accessible from any device
- ✅ 95% of desktop code reused

---

**Ready to implement?** Next: Create backend API structure!

