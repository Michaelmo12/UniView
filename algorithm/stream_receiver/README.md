# Stream Receiver Service

A microservice that receives synchronized video streams from 8 drone streamers and displays them in a real-time 2×4 grid.

## Overview

This service acts as a **stream receiver and visualizer** that:

- Listens for incoming TCP connections from 8 drone streamers
- Receives binary packets containing frames + calibration data
- Synchronizes frames across all 8 drones by frame number
- Displays synchronized frames in a 2×4 OpenCV grid
- Shows camera calibration metadata overlays (intrinsic, extrinsic)

## Architecture

```
Stream Receiver Service
├── client.py              # Main receiver entry point
├── src/
│   ├── protocol.py        # Binary protocol (unpack_packet, recv_exact)
│   └── sync_buffer.py    # Frame synchronization buffer
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Features

- **8 Concurrent Receivers**: Each drone has its own listener thread
- **TCP Server Mode**: Listens on ports 15000-15007
- **Frame Synchronization**: FrameBuffer ensures all 8 drones show same frame_num
- **Real-Time Display**: 2×4 grid at 2 FPS (1920×540 resolution)
- **Calibration Overlay**: Shows fx, fy, cx, cy, position (x, y, z)
- **Thread-Safe**: Lock-based synchronization for buffer access
- **Graceful Shutdown**: ESC key or Ctrl+C stops cleanly

## Display Layout

```
┌─────────┬─────────┬─────────┬─────────┐
│ Drone 0 │ Drone 1 │ Drone 2 │ Drone 3 │
│ 480×270 │ 480×270 │ 480×270 │ 480×270 │
├─────────┼─────────┼─────────┼─────────┤
│ Drone 4 │ Drone 5 │ Drone 6 │ Drone 7 │
│ 480×270 │ 480×270 │ 480×270 │ 480×270 │
└─────────┴─────────┴─────────┴─────────┘
Total: 1920×540 pixels
```

Each cell shows:
- **Line 1**: Drone ID and frame number (green)
- **Line 2**: Intrinsic parameters (fx, fy)
- **Line 3**: Principal point (cx, cy)
- **Line 4**: Camera position (x, y, z) (yellow)

## Requirements

### Software
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

### Network
- Ports 15000-15007 available (localhost only)

## Installation

1. **Navigate to directory**:
   ```bash
   cd algorithm/stream_receiver
   ```

2. **Activate virtual environment**:
   ```bash
   # Windows
   ..\..\venv\Scripts\activate

   # Linux/Mac
   source ../../venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Important: Start Order

⚠️ **START THIS RECEIVER FIRST**, then start the mock drone streamer service.

This service must be listening before the streamer attempts to connect.

### Running the Receiver

```bash
python client.py
```

### Expected Output

```
======================================================================
📺 STREAM RECEIVER SERVICE - Receiver and Display
======================================================================
Configuration:
  Drones: 8
  Ports: 15000-15007
  Protocol: Custom Binary TCP
  Socket timeout: 5.0s
======================================================================

⚠️  NOTE: Start this receiver FIRST, then start the drone streamer

Starting 8 receiver threads...
📺 Drone 0: Listening on port 15000
📺 Drone 1: Listening on port 15001
...
✅ All 8 receivers listening

📺 Drone 0: Waiting for connection...
📺 Drone 1: Waiting for connection...
...

(After starting the streamer service)

📺 Drone 0: Connected to ('127.0.0.1', 50123)
📺 Drone 1: Connected to ('127.0.0.1', 50124)
...

======================================================================
🎥 Display Active
======================================================================
Press ESC to quit

⏳ Waiting for synchronized frames from all 8 drones...
✅ Displayed synchronized frame 0000 (FPS: 2.0)
📥 Drone 0: Received frame 0100
✅ Displayed synchronized frame 0100 (FPS: 2.1)
...
```

### Stopping the Service

**Option 1**: Press **ESC** in the OpenCV window:
```
⚠️  User requested quit (ESC pressed)
```

**Option 2**: Press **Ctrl+C** in terminal:
```
⚠️  Interrupted by user
```

Both methods shutdown gracefully:
```
Display stopped - 1000 frames displayed

Final Statistics:
  Frames synchronized: 1000
  Drone 0: 1000 received, 0 dropped
  Drone 1: 1000 received, 0 dropped
  ...
✅ Client stopped
```

## Configuration

Edit [client.py](client.py) to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_DRONES` | `8` | Number of drones to receive |
| `BASE_PORT` | `15000` | Starting port (increments by 1 per drone) |
| `SOCKET_TIMEOUT` | `5.0` | Receive timeout (seconds) |
| `max_buffer_size` | `10` | Max frames buffered per drone |

## Frame Synchronization

### How It Works

The `FrameBuffer` class ensures perfect synchronization:

1. **Receive**: Each receiver thread adds frames to buffer
2. **Buffer**: Stores frames by `{drone_id: {frame_num: data}}`
3. **Scan**: Display loop scans for frame_num where all 8 drones have data
4. **Display**: When complete set found, display and advance to next frame
5. **Cleanup**: Remove old frames to prevent memory leak

### Example

```
current_frame = 42

Buffer state:
  Drone 1: {42, 43, 44}
  Drone 2: {42, 43}
  Drone 3: {42, 43, 44}
  ...
  Drone 7: {42, 43}
  Drone 8: {42, 43}

Result: get_synchronized_set() returns (42, {1-8: data})
```

After display, `current_frame` advances to 43 and frames < 43 are removed.

## API / Integration

### Connection Model

- **Protocol**: TCP (SOCK_STREAM)
- **Mode**: Server (listens for connections)
- **Ports**: 15000-15007 (one per drone)
- **Host**: localhost (127.0.0.1)

### Packet Protocol

Packets received are unpacked using [src/protocol.py](src/protocol.py):

```python
from src.protocol import unpack_packet, recv_exact, HEADER_SIZE, CALIBRATION_SIZE

# Read header
header = recv_exact(sock, HEADER_SIZE)
drone_id, frame_num, jpeg_size, timestamp = struct.unpack('B I I d', header)

# Read JPEG
jpeg_bytes = recv_exact(sock, jpeg_size)

# Read calibration
calib_bytes = recv_exact(sock, CALIBRATION_SIZE)

# Reconstruct and unpack
packet = header + jpeg_bytes + calib_bytes
drone_id, frame_num, frame, K, R, t, dist, timestamp = unpack_packet(packet)
```

### Frame Buffer

Use `FrameBuffer` from [src/sync_buffer.py](src/sync_buffer.py):

```python
from src.sync_buffer import FrameBuffer

buffer = FrameBuffer(num_drones=8, max_buffer_size=10)

# Add frame (from receiver thread)
buffer.add_frame(drone_id, frame_num, {
    'frame': frame,
    'K': K,
    'R': R,
    't': t,
    'dist': dist,
    'timestamp': timestamp
})

# Get synchronized set (from display thread)
frame_num, frames_dict = buffer.get_synchronized_set()
if frames_dict is not None:
    # All 8 drones have this frame_num
    display(frames_dict)
```

## Troubleshooting

### Problem: "No connection after 10s"

**Cause**: Streamer service not started

**Solution**:
1. Start this receiver first and wait for "All 8 receivers listening"
2. Start the streamer service: `cd ../mock_drone_streamer && python server.py`

---

### Problem: "Timeout (no data for 5.0s)"

**Cause**: Streamer stopped sending or crashed

**Solution**:
1. Check streamer service terminal for errors
2. Restart both services (receiver first, then streamer)

---

### Problem: Frames not synchronized / waiting forever

**Cause**: One or more drones not sending data

**Solution**:
1. Check buffer status in terminal output:
   ```
   ⏳ Waiting... (current_frame=42, buffer_sizes={0: 3, 1: 0, 2: 3, ...})
   ```
2. Identify which drone has `buffer_size=0` (not receiving)
3. Check streamer logs for that drone ID

---

### Problem: Display is choppy or laggy

**Cause**: High CPU usage or slow rendering

**Solution**:
1. Close other applications
2. Reduce grid cell size in [client.py](client.py:150-151):
   ```python
   cell_width = 320  # instead of 480
   cell_height = 180  # instead of 270
   ```
3. Check FPS counter in display - should be ~2 FPS

---

### Problem: "Connection lost" errors

**Cause**: Streamer crashed or network issue

**Solution**:
1. Check streamer service for errors
2. Restart both services
3. On Windows, ensure firewall allows Python on ports 15000-15007

## Performance

| Metric | Value |
|--------|-------|
| **Receive Rate** | 2 FPS per drone |
| **Display Rate** | 2 FPS synchronized |
| **Network Bandwidth** | ~3.5 MB/s (8 drones) |
| **CPU Usage** | 10-20% (modern CPU) |
| **Memory** | ~500 MB (frame buffering) |
| **Latency** | <10ms (localhost TCP) |

## Advanced

### Custom Display Layout

Modify `create_grid_display()` in [client.py](client.py:135) to change:

- Grid dimensions (rows, columns)
- Cell size (resolution per drone)
- Overlay text (what to display)
- Text color, font, position

### Export to Video

Add video writer after line 289:

```python
# At start of display_loop()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output.mp4', fourcc, 2.0, (1920, 540))

# In display loop (after cv2.imshow)
video_writer.write(grid)

# At end
video_writer.release()
```

## License

Internal project for UniView system. Not for public distribution.

## Contact

For issues or questions, contact: מיכאל מורדכי (Michael Mordechai)
