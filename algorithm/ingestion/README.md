# Ingestion Stage

This stage receives and structures data from the TCP drone streams.

## Files

### `models.py`

Defines the core data structures used throughout the pipeline:

#### **CameraCalibration**
Complete camera parameters for a single frame:
- **K** (3x3): Intrinsic matrix - focal length and optical center
- **R** (3x3): Rotation matrix - camera orientation in world space
- **t** (3x1): Translation vector - camera position
- **dist** (5,): Distortion coefficients - lens correction parameters

Properties:
- `projection_matrix`: P = K @ [R | t] - linear transformation from 3D world → 2D pixels
- `camera_center`: Camera position in world coordinates

#### **DroneFrame**
Single frame from one drone:
- `drone_id` (1-8): Which drone
- `frame_num` (0-999): Sequence number
- `timestamp`: When captured
- `frame`: BGR image (H, W, 3)
- `calibration`: CameraCalibration for this frame

#### **SynchronizedFrameSet**
Frames from multiple drones at the same timestamp:
- `frames`: Dict {drone_id → DroneFrame}
- `is_complete`: True if all 8 drones present
- `missing_drones`: List of missing drone IDs

Used to group frames for cross-camera matching.

## Usage

```python
from ingestion.models import CameraCalibration, DroneFrame, SynchronizedFrameSet

# TCP receiver creates DroneFrame objects
# Synchronizer groups them into SynchronizedFrameSet
# Detection stage processes each synchronized set
```
