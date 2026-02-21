# Algorithm Pipeline Testing Guide

This guide explains how to test the complete algorithm pipeline using real MATRIX dataset frames streamed from the mock drone server.

## Pipeline Stages

The current pipeline processes frames through these stages:
1. **Ingestion** - TCP receiver + synchronizer (receives frames from 8 drones, groups by frame number)
2. **Detection** - YOLO person detection
3. **Features** - WCH appearance descriptor extraction
4. **Fusion** - Cross-camera matching (epipolar geometry + appearance)
5. **[Future Phase 3]** - 3D Reconstruction
6. **[Future Phase 4]** - Temporal Tracking
7. **[Future Phase 5]** - WebSocket Output

## End-to-End Testing

### Prerequisites

1. MATRIX dataset available at: `C:\Projects_H.W\FINAL-PROJECT\UniView\MATRIX_30x30`
2. YOLO weights at: `algorithm/weights/best.pt`
3. Python environment activated: `venv\Scripts\activate`

### Step 1: Start the Mock Drone Streamer

The mock drone streamer simulates 8 drones streaming frames via TCP.

```bash
# Terminal 1: Start the streamer
cd C:\Projects_H.W\FINAL-PROJECT\UniView\mock_drone_streamer
python server.py
```

**Expected Output:**
```
======================================================================
MOCK DRONE STREAMER SERVICE
======================================================================
Configuration:
  Dataset: C:\Projects_H.W\FINAL-PROJECT\UniView\MATRIX_30x30
  Drones: 8
  Ports: 15000-15007
  Frames: 30 per drone
  Stream rate: 2 FPS (500ms per frame)
======================================================================

[DRONE 1] Starting on port 15000...
[DRONE 2] Starting on port 15001...
...
[DRONE 8] Ready (8/8)
======================================================================
[SERVER] All 8 drones ready! Starting stream in 1 second...
======================================================================
```

The streamer will wait for clients to connect and start streaming when clients are ready.

### Step 2: Run the Pipeline Test

The pipeline test connects to all 8 drone streams via TCP, processes frames through all stages, and validates the integration.

**IMPORTANT**: The streamer must be running first! The pipeline connects as a TCP client.

```bash
# Terminal 2: Run the pipeline (while streamer is running in Terminal 1)
cd C:\Projects_H.W\FINAL-PROJECT\UniView\algorithm
python run_pipeline.py --num-frames 5

# Or test with fewer drones (useful for faster testing):
python run_pipeline.py --num-frames 3 --num-drones 3
```

**What Happens:**

1. **TCP Receivers** connect to ports 15000-15007
2. **Protocol Decoder** parses binary packets into DroneFrames
3. **Synchronizer** groups frames by frame_num into SynchronizedFrameSets
4. **Batch Detector** runs YOLO on all frames
5. **WCH Extractor** extracts appearance features from detected persons
6. **Cross-Camera Matcher** fuses detections across cameras

**Expected Output:**
```
================================================================================
ALGORITHM PIPELINE RUNNER
================================================================================

Pipeline stages:
  1. Ingestion → SynchronizedFrameSet
  2. Detection → YOLO person detection
  3. Features → WCH extraction
  4. Fusion → Cross-camera matching

Initializing pipeline stages...
  ✓ Detection: YOLOv11 on cpu
  ✓ Features: WCH 16 bins/channel
  ✓ Fusion: epipolar<5.0px, appearance>0.7
Pipeline initialized successfully

================================================================================
Processing Frame 0
================================================================================
Stage 1: Ingestion - 8 drones
  Drone 1: (1080, 1920, 3) @ 0.000s
  Drone 2: (1080, 1920, 3) @ 0.000s
  ...
  Drone 8: (1080, 1920, 3) @ 0.000s

Stage 2: Detection - 12 persons in 1850.2ms
  Drone 1: 2 detections
  Drone 2: 3 detections
  ...

Stage 3: Features - 12/12 features in 45.3ms

Stage 4: Fusion - 3 groups, 2 unmatched in 15.7ms
  Group 1: 3 cameras [1, 2, 4], 3 matches
    Drone 1 <-> Drone 2: epipolar=2.34px, appearance=0.85
    Drone 2 <-> Drone 4: epipolar=3.12px, appearance=0.78
    Drone 1 <-> Drone 4: epipolar=4.56px, appearance=0.72
  Group 2: 2 cameras [3, 5], 1 matches
    Drone 3 <-> Drone 5: epipolar=1.89px, appearance=0.91
  ...

Total pipeline time: 1911.2ms
  Detection: 96.8% | Features: 2.4% | Fusion: 0.8%

================================================================================
PIPELINE SUMMARY
================================================================================
Frames processed: 5
Total detections: 58
Total features extracted: 58
Total match groups: 14
Average pipeline time: 1895.4ms/frame

✓ Pipeline integration successful!
```

### Step 3: Validation

The test validates:

- **Data Flow**: Frames flow correctly from ingestion → detection → features → fusion
- **Metadata Preservation**: drone_id, frame_num, local_id preserved across stages
- **Feature Integration**: Detection.features populated by WCH extractor
- **Fusion Integration**: FusionResult uses features from previous stage
- **Cross-Stage Dependencies**: Projection matrices flow from ingestion to fusion

### Alternative: Test with Synthetic Data

If you don't have the MATRIX dataset or want a quick test:

```bash
python run_pipeline.py --test-synthetic --num-frames 2
```

This generates synthetic multi-camera frames with known ground truth (Person A in cameras 1-2, Person B in cameras 2-3, Person C only in camera 1).

## Troubleshooting

### "Connection refused" on ports 15000-15007

- Make sure the mock_drone_streamer is running first
- Check firewall settings

### "No YOLO weights found"

```bash
# Check weights exist
ls algorithm/weights/best.pt

# If missing, train or download weights first
cd AI/scripts
python train_baseline.py
```

### "No detections found"

- YOLO might not detect persons in some frames
- Try different frames: `--num-frames 10`
- Check YOLO confidence threshold in [config/settings.py](config/settings.py)

### Slow performance

- Detection is CPU-intensive (~2s/frame on CPU)
- For faster testing: `--num-frames 1`
- For GPU: install CUDA-enabled PyTorch

## Next Steps

As new pipeline stages are added:

1. **Phase 3 (Reconstruction)**: Add 3D triangulation after fusion
2. **Phase 4 (Tracking)**: Add temporal tracking after reconstruction
3. **Phase 5 (Output)**: Add WebSocket streaming of final results

Update `run_pipeline.py` to include new stages in the processing flow and validation checks.

## Integration Test Files

- [run_pipeline.py](run_pipeline.py) - Main pipeline runner
- [test_pipeline_integration.py](test_pipeline_integration.py) - Synthetic data integration test
- [ingestion/tcp_receiver.py](ingestion/tcp_receiver.py) - TCP client for mock drone streams
- [ingestion/synchronizer.py](ingestion/synchronizer.py) - Frame synchronization
- [detection/batch_detector.py](detection/batch_detector.py) - YOLO detection orchestrator
- [features/wch_extractor.py](features/wch_extractor.py) - WCH feature extraction
- [fusion/cross_camera_matcher.py](fusion/cross_camera_matcher.py) - Cross-camera fusion orchestrator
