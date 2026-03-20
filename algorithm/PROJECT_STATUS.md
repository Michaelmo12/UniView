# UniView Algorithm Pipeline - Project Status

**Last Updated:** 2026-02-03
**Current Phase:** Ingestion + Detection Complete, Starting Fusion

---

## ✅ COMPLETED MODULES

### 1. Ingestion Stage (`ingestion/`)
**Purpose:** Receive and synchronize frames from 8 drone TCP streams

**Files:**
- `models.py` - Data structures (DroneFrame, CameraCalibration, SynchronizedFrameSet)
- `protocol_decoder.py` - Binary packet parsing (JPEG + calibration)
- `tcp_receiver.py` - TCP client receiving frames from mock_drone_streamer
- `synchronizer.py` - Groups frames by frame_num, outputs complete sets
- `__init__.py` - Exports all components

**Status:** ✅ **FULLY WORKING** - Tested with real streamer, 91.7% complete sets

**Key Design:**
- Each drone gets own TCPReceiver thread
- Synchronizer waits for all 8 drones OR 200ms timeout
- Uses logging instead of print()
- All settings from `config.settings`

---

### 2. Detection Stage (`detection/`)
**Purpose:** Run YOLO person detection on synchronized frames

**Files:**
- `models.py` - BoundingBox, Detection, DetectionSet structures
- `yolo_detector.py` - YOLOv8 wrapper for single-frame detection
- `batch_detector.py` - Process entire SynchronizedFrameSet (all 8 frames)
- `__init__.py` - Exports all components

**Status:** ✅ **IMPLEMENTED** - Code ready, not fully tested yet

**Key Design:**
- Uses custom weights: `weights/best.pt`
- One YOLO instance shared across all frames
- Returns `{drone_id: DetectionSet}` for each synchronized set
- All settings from `config.settings.detection`

---

### 3. Configuration (`config/`)
**Purpose:** Centralized settings for all modules

**Files:**
- `settings.py` - NetworkConfig, IngestionConfig, DetectionConfig
- `__init__.py` - Exports settings singleton

**Usage:**
```python
from config import settings

# Network
host = settings.network.host              # "127.0.0.1"
port = settings.network.base_port         # 15000

# Ingestion
num_drones = settings.ingestion.num_drones           # 8
sync_timeout = settings.ingestion.sync_timeout       # 0.2

# Detection
weights = settings.weights_path                      # weights/best.pt
confidence = settings.detection.conf_threshold       # 0.5
device = settings.detection.device                   # "cuda"
```

**Status:** ✅ **COMPLETE** - All modules use config

---

## 🚧 TODO: REMAINING MODULES

### 4. Fusion Stage (`fusion/`) - **NEXT TO BUILD**
**Purpose:** Match detections across cameras using epipolar geometry

**Planned Files:**
- `models.py` - CrossCameraMatch, MatchSet
- `fundamental_matrix.py` - Compute F matrix between camera pairs
- `epipolar_filter.py` - Filter matches using epipolar constraint
- `visual_matcher.py` - Appearance-based matching (optional)
- `cross_drone_matcher.py` - Main orchestrator

**Key Concepts:**
- Epipolar geometry: Given detection in camera A, find corresponding point in camera B
- Fundamental matrix F: Relates corresponding points between two views
- Epipolar line: Constraint line where corresponding point must lie
- Match validation: Distance from point to epipolar line < threshold

**Inputs:** `{drone_id: DetectionSet}` from BatchDetector
**Outputs:** `MatchSet` - Groups of detections that represent same person across cameras

---

### 5. Reconstruction Stage (`reconstruction/`) - **AFTER FUSION**
**Purpose:** Triangulate 3D positions from matched 2D detections

**Planned Files:**
- `models.py` - Point3D, PersonReconstruction
- `midpoint_triangulation.py` - Triangulate from 2+ views
- `dbscan_clustering.py` - Cluster 3D points (handles noise)

**Key Concepts:**
- Triangulation: Given 2+ views of same point, compute 3D position
- Midpoint method: Find closest point between viewing rays
- DBSCAN: Cluster points, remove outliers

**Inputs:** `MatchSet` from Fusion
**Outputs:** List of `Point3D` (x, y, z positions of people)

---

### 6. Tracking Stage (`tracking/`) - **AFTER RECONSTRUCTION**
**Purpose:** Track people across frames with global IDs

**Planned Files:**
- `models.py` - Track, TrackState
- `kalman_filter.py` - Predict next position, smooth trajectory
- `track_associator.py` - Match detections to existing tracks
- `global_id_manager.py` - Assign persistent IDs

**Key Concepts:**
- Kalman filter: Predict + correct for smooth tracking
- Hungarian algorithm: Optimal detection-to-track assignment
- Global IDs: Same person gets same ID across frames

**Inputs:** List of `Point3D` per frame
**Outputs:** List of `Track` with global IDs and trajectories

---

### 7. Pipeline Orchestrator (`pipeline/`) - **FINAL INTEGRATION**
**Purpose:** Connect all stages into complete pipeline

**Planned Files:**
- `inference_pipeline.py` - Main pipeline thread
- `stage_connector.py` - Queue-based stage connections
- `output_formatter.py` - Format results for API/visualization

**Flow:**
```
TCP Streams → Ingestion → Detection → Fusion → Reconstruction → Tracking → Output
```

---

## 📋 KEY ARCHITECTURE DECISIONS

### 1. **Config-Only Constructors**
**Rule:** No parameters in `__init__`, all from `config.settings`

```python
# ✅ CORRECT
receiver = TCPReceiver(drone_id=1, output_queue=queue)

# ❌ WRONG (don't pass host/port)
receiver = TCPReceiver(drone_id=1, host="...", port=..., output_queue=queue)
```

### 2. **Logging, Not Print**
**Rule:** Always use `logging.getLogger(__name__)`, never `print()`

```python
# ✅ CORRECT
logger = logging.getLogger(__name__)
logger.info("Frame %d received", frame_num)

# ❌ WRONG
print(f"Frame {frame_num} received")
```

### 3. **No Inline If-Else**
**Rule:** Use traditional if statements, not ternary operators

```python
# ✅ CORRECT
if value is None:
    result = default
else:
    result = value

# ❌ WRONG
result = default if value is None else value
```

### 4. **Synchronizer Strategy**
**Rule:** Wait for ALL drones OR timeout (no partial threshold)

```python
# Simple rule:
if num_drones_present == 8:
    output_immediately()
# else: wait for timeout (200ms), then output partial set
```

---

## 🗂️ PROJECT STRUCTURE

```
algorithm/
├── config/
│   ├── settings.py          ✅ Network, Ingestion, Detection configs
│   └── __init__.py
├── ingestion/
│   ├── models.py            ✅ DroneFrame, CameraCalibration, SynchronizedFrameSet
│   ├── protocol_decoder.py ✅ Binary packet parsing
│   ├── tcp_receiver.py      ✅ TCP client + threading
│   ├── synchronizer.py      ✅ Frame synchronization
│   └── __init__.py
├── detection/
│   ├── models.py            ✅ BoundingBox, Detection, DetectionSet
│   ├── yolo_detector.py     ✅ YOLOv8 wrapper
│   ├── batch_detector.py    ✅ Process SynchronizedFrameSet
│   └── __init__.py
├── fusion/                  🚧 NEXT TO BUILD
│   ├── models.py
│   ├── fundamental_matrix.py
│   ├── epipolar_filter.py
│   ├── visual_matcher.py
│   └── cross_drone_matcher.py
├── reconstruction/          🚧 AFTER FUSION
│   ├── models.py
│   ├── midpoint_triangulation.py
│   └── dbscan_clustering.py
├── tracking/                🚧 AFTER RECONSTRUCTION
│   ├── models.py
│   ├── kalman_filter.py
│   ├── track_associator.py
│   └── global_id_manager.py
├── pipeline/                🚧 FINAL INTEGRATION
│   ├── inference_pipeline.py
│   ├── stage_connector.py
│   └── output_formatter.py
├── weights/
│   └── best.pt              ✅ Custom YOLO weights (19MB)
└── main.py                  🚧 Entry point
```

---

## 🧪 TESTING

### Test Ingestion + Synchronization:
```bash
# Terminal 1: Start streamer
cd mock_drone_streamer
python server.py

# Terminal 2: Test synchronizer
cd algorithm
python ingestion/synchronizer.py
```

**Expected:** 91-100% complete sets (all 8 drones synchronized)

### Test Detection:
```bash
# With streamer running:
cd algorithm
python -m detection.batch_detector
```

**Expected:** Detections from 3 drones, ~0.5s inference time per frame

---

## 📝 NEXT STEPS (IN ORDER)

1. **Build Fusion Stage** (epipolar geometry, cross-camera matching)
2. **Build Reconstruction Stage** (triangulation, 3D positioning)
3. **Build Tracking Stage** (Kalman filter, global IDs)
4. **Build Pipeline Orchestrator** (connect all stages)
5. **Build API/WebSocket** (serve results to frontend)
6. **Integration Testing** (full end-to-end pipeline)

---

## 🔑 KEY COMMANDS

### Run Configuration:
```bash
cd algorithm
python config/settings.py
```

### Test Individual Modules:
```bash
python ingestion/models.py
python ingestion/protocol_decoder.py
python ingestion/tcp_receiver.py
python ingestion/synchronizer.py
python detection/models.py
python detection/yolo_detector.py
python -m detection.batch_detector
```

### Import From Other Modules:
```python
from ingestion import DroneFrame, SynchronizedFrameSet, TCPReceiver, FrameSynchronizer
from detection import BoundingBox, Detection, DetectionSet, BatchDetector
from config import settings
```

---

## ⚠️ IMPORTANT NOTES

1. **Always start mock_drone_streamer first** before running tests
2. **Ports:** Drone 1→15000, Drone 2→15001, ..., Drone 8→15007
3. **YOLO weights:** Must exist at `algorithm/weights/best.pt`
4. **Python path:** Run from `algorithm/` directory or use `python -m module.name`
5. **Config is king:** All settings centralized, no hardcoded values

---

## 🎯 PROJECT GOAL

**Input:** 8 synchronized TCP video streams from drones
**Output:** Real-time 3D positions + global IDs of all people in scene
**Use Case:** Multi-camera person tracking and 3D reconstruction
