# Patch Clamp Electrophysiology App - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ Voltage  │ │ Current  │ │ Protocol │ │  Batch   │         │
│  │  Clamp   │ │  Clamp   │ │  Editor  │ │ Analysis │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │   IV     │ │  Event   │ │ Statistics│ │    QC    │         │
│  │  Curves  │ │Detection │ │           │ │          │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS ENGINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Voltage    │  │   Current    │  │    Event     │         │
│  │   Clamp      │  │   Clamp      │  │  Detection   │         │
│  │  Analysis    │  │  Analysis    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Kinetic    │  │ Statistical  │  │   Quality    │         │
│  │   Analysis   │  │   Analysis   │  │   Control    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Filtering   │  │  Baseline    │  │   Artifact   │         │
│  │              │  │  Correction  │  │   Removal    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Curve      │  │   Signal     │  │   Noise      │         │
│  │   Fitting    │  │  Processing  │  │   Analysis   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA I/O LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │   ATF    │ │   ABF    │ │   HEKA   │ │   Neo    │           │
│  │ Handler  │ │ Handler  │ │ Handler  │ │ Handler  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │   CSV    │ │  Binary  │ │  Format  │                       │
│  │ Handler  │ │ Handler  │ │ Detector │                       │
│  └──────────┘ └──────────┘ └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   File       │  │  Database    │  │   Metadata   │         │
│  │   System     │  │  (SQLite)    │  │   Manager    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
app.py (Main Application)
    │
    ├─── GUI Components
    │    ├─── voltage_clamp_tab.py
    │    ├─── current_clamp_tab.py
    │    ├─── protocol_editor_tab.py
    │    ├─── iv_curve_tab.py
    │    ├─── event_detection_tab.py
    │    └─── statistics_tab.py
    │
    ├─── Analysis Modules
    │    ├─── voltage_clamp/
    │    │    ├─── iv_curve_analyzer.py
    │    │    ├─── activation_kinetics.py
    │    │    └─── leak_subtraction.py
    │    │
    │    ├─── current_clamp/
    │    │    ├─── action_potential_detector.py
    │    │    ├─── membrane_properties.py
    │    │    └─── firing_patterns.py
    │    │
    │    ├─── events/
    │    │    ├─── synaptic_event_detector.py
    │    │    └─── event_characterizer.py
    │    │
    │    └─── kinetics/
    │         ├─── exponential_fitting.py
    │         └─── boltzmann_fitting.py
    │
    ├─── Data Processing
    │    ├─── filtering/
    │    ├─── baseline_correction.py
    │    └─── artifact_removal.py
    │
    ├─── I/O Layer
    │    ├─── io_utils/
    │    │    ├─── atf_handler.py (existing)
    │    │    ├─── abf_handler.py (NEW)
    │    │    ├─── heka_handler.py (NEW)
    │    │    └─── format_detector.py (NEW)
    │    │
    │    └─── protocols/
    │         ├─── protocol_editor.py (NEW)
    │         └─── protocol_executor.py (NEW)
    │
    └─── Data Management
         ├─── database.py (NEW)
         └─── metadata_manager.py (NEW)
```

## Data Flow

### Loading Data
```
File → Format Detector → Appropriate Handler → Data Object → Analysis Engine
```

### Analysis Pipeline
```
Raw Data
    ↓
[Filtering]
    ↓
[Baseline Correction]
    ↓
[Artifact Removal]
    ↓
[Analysis Module Selection]
    ├──→ Voltage Clamp Analysis
    ├──→ Current Clamp Analysis
    └──→ Event Detection
    ↓
[Results]
    ↓
[Visualization]
    ↓
[Export/Storage]
```

## Key Data Structures

### Experiment Data
```python
@dataclass
class ExperimentData:
    """Core data structure for electrophysiology experiments"""
    # Raw data
    time: np.ndarray
    voltage: Optional[np.ndarray]  # For voltage clamp
    current: Optional[np.ndarray]   # For current clamp
    
    # Metadata
    sampling_rate: float
    units: Dict[str, str]  # {'voltage': 'mV', 'current': 'pA'}
    protocol: ProtocolInfo
    metadata: Dict[str, Any]
    
    # Processed data
    filtered_data: Optional[np.ndarray]
    baseline_corrected: Optional[np.ndarray]
    
    # Analysis results
    analysis_results: Dict[str, Any]
```

### Protocol Definition
```python
@dataclass
class Protocol:
    """Protocol definition structure"""
    name: str
    epochs: List[Epoch]
    repetitions: int
    inter_sweep_interval: float
    
@dataclass
class Epoch:
    """Single epoch in a protocol"""
    type: EpochType  # STEP, RAMP, SINE, etc.
    duration: float
    amplitude: float
    holding: Optional[float]
```

### Analysis Results
```python
@dataclass
class AnalysisResult:
    """Standardized analysis result structure"""
    analysis_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    timestamp: datetime
```

## Integration Strategy

### Phase 1: Foundation
1. Add file format support (ABF first)
2. Create data structure abstractions
3. Enhance existing analysis modules

### Phase 2: Core Features
1. Complete current clamp analysis
2. Add IV curve analysis
3. Implement event detection

### Phase 3: Advanced Features
1. Protocol editor
2. Batch processing
3. Statistical tools

### Phase 4: Polish
1. UI improvements
2. Documentation
3. Performance optimization

## Extension Points

### Plugin System
```python
class AnalysisPlugin:
    """Base class for analysis plugins"""
    def analyze(self, data: ExperimentData) -> AnalysisResult:
        raise NotImplementedError
    
    def get_ui_component(self) -> tk.Widget:
        raise NotImplementedError
```

### Custom Filters
```python
class CustomFilter:
    """Base class for custom filters"""
    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

### Export Formats
```python
class ExportHandler:
    """Base class for export handlers"""
    def export(self, data: ExperimentData, path: str):
        raise NotImplementedError
```

## Performance Considerations

### Memory Management
- Lazy loading of large datasets
- Streaming for real-time acquisition
- Efficient data structures (numpy arrays)

### Processing Speed
- Vectorized operations (numpy)
- Parallel processing for batch operations
- Caching of intermediate results

### Scalability
- Database for large datasets
- Efficient querying
- Incremental loading

## Testing Strategy

### Unit Tests
- Individual analysis functions
- File format handlers
- Data processing modules

### Integration Tests
- End-to-end analysis pipelines
- Multi-format loading
- Batch processing

### Performance Tests
- Large file handling
- Real-time processing
- Memory usage

## Documentation Structure

```
docs/
├── user_guide/
│   ├── getting_started.md
│   ├── voltage_clamp.md
│   ├── current_clamp.md
│   └── protocols.md
├── api_reference/
│   ├── analysis_modules.md
│   ├── data_structures.md
│   └── plugin_development.md
└── examples/
    ├── basic_analysis.py
    ├── batch_processing.py
    └── custom_analysis.py
```

