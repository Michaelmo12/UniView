# ENet Drone Streamer Microservice

A standalone mock drone streaming microservice that transmits MATRIX dataset frames
over ENet (reliable UDP).  Provides a drop-in replacement for the TCP-based
`mock_drone_streamer` with lower latency and no head-of-line blocking.

---

## Architecture

```
MATRIX Dataset (disk)
        |
        v
  DatasetLoader          per-drone process
  (frames/ + calib/)
        |
        v
  PacketBuilder          binary packet assembly
  (13B header + 104B calib + JPEG)
        |
        v
  ENetStreamer           ENet UDP server
  (pyenet host, channel 0, RELIABLE)
        |  ... ENet packets ...
        v
  ENetReceiver           inside algorithm/src/ingestion/
  (parses DroneFrame)
        |
        v
  Ingestion Pipeline     existing synchronizer + stages
```

One process per drone.  Each drone listens on a unique port:
- Drone 1 → `16000`, Drone 2 → `16001`, ..., Drone 8 → `16007`

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Requirements:
- `pyenet>=1.3` — Python bindings for the ENet reliable-UDP library
- `numpy` — Array math for calibration matrices
- `opencv-python` — JPEG encode/decode and synthetic frame generation

---

## Usage

### Single drone

```bash
# Stream Drone 1 frames from MATRIX dataset at 2 FPS
python main.py --drone-id 1 --dataset /path/to/MATRIX

# Custom FPS and JPEG quality
python main.py --drone-id 2 --dataset /path/to/MATRIX --fps 5 --jpeg-quality 70

# Bind on a specific interface
python main.py --drone-id 1 --dataset /path/to/MATRIX --host 192.168.1.10

# Override the port (default: 16000 + drone_id - 1)
python main.py --drone-id 1 --dataset /path/to/MATRIX --port 17000

# Stop after one pass through the dataset (no looping)
python main.py --drone-id 1 --dataset /path/to/MATRIX --no-loop

# Verbose debug logging
python main.py --drone-id 1 --dataset /path/to/MATRIX --debug
```

### All 8 drones

```bash
# Launch all 8 drone processes (ports 16000-16007)
python run_all_drones.py --dataset /path/to/MATRIX

# Custom frame rate
python run_all_drones.py --dataset /path/to/MATRIX --fps 5

# Press Ctrl+C to stop all drones cleanly
```

---

## Packet Format

All values are packed **little-endian**.

### Header (13 bytes)

| Offset | Size | Type   | Field         | Description                          |
|--------|------|--------|---------------|--------------------------------------|
| 0      | 1    | uint8  | `drone_id`    | Drone identifier (1–8)               |
| 1      | 4    | uint32 | `frame_num`   | Frame sequence number                |
| 5      | 8    | uint64 | `timestamp_ns`| Capture time (nanoseconds since epoch)|

Struct format: `<BIQ`

### Calibration Block (104 bytes)

| Offset | Size | Type      | Field  | Description                            |
|--------|------|-----------|--------|----------------------------------------|
| 13     | 36   | 9×float32 | `K`    | Intrinsic matrix 3×3, row-major        |
| 49     | 36   | 9×float32 | `R`    | Rotation matrix 3×3, row-major         |
| 85     | 12   | 3×float32 | `t`    | Translation vector 3×1, flattened      |
| 97     | 20   | 5×float32 | `dist` | Distortion coefficients [k1,k2,p1,p2,k3]|

Struct format: `<9f9f3f5f`

### Payload (variable)

| Offset | Size     | Type  | Field       | Description               |
|--------|----------|-------|-------------|---------------------------|
| 117    | variable | bytes | `jpeg_data` | JPEG-encoded BGR frame    |

**Total fixed portion:** 13 + 104 = **117 bytes**

---

## MATRIX Dataset Structure

Expected layout for the `--dataset` path:

```
MATRIX/
    Drone1/
        frames/
            frame_0000.jpg
            frame_0001.jpg
            ...
        calibration/
            K.txt                  <- 3x3 intrinsic (3 rows, space-separated)
            R_frame_0000.txt       <- 3x3 rotation per frame
            t_frame_0000.txt       <- 3x1 translation per frame (3 lines, 1 value each)
    Drone2/
        ...
    ...
    Drone8/
        ...
```

**Distortion:** MATRIX does not provide distortion coefficients. All `dist` values
are set to `[0, 0, 0, 0, 0]` automatically.

---

## Synthetic Fallback

If the `--dataset` path does not exist or a drone folder is missing, the loader
falls back to **synthetic data automatically** — no error is raised:

- Frames: 640×480 solid-color images with drone ID and frame number text overlay
- Intrinsics: Identity-style K with fx=fy=800, cx=320, cy=240
- Extrinsics: Identity R, zero t
- Distortion: Zero dist

This allows the streamer to run and the receiver to exercise the full packet
parsing pipeline without a dataset.

```bash
# Works even with no MATRIX dataset
python main.py --drone-id 1 --dataset /nonexistent/path
```

---

## ENetReceiver (Algorithm Side)

The matching receiver lives at `algorithm/src/ingestion/enet_receiver.py`.
It has the same public API as `TCPReceiver`:

```python
from algorithm.src.ingestion.enet_receiver import ENetReceiver
import queue

q = queue.Queue()
receiver = ENetReceiver(drone_id=1, output_queue=q)
receiver.start()

frame = q.get(timeout=10.0)  # DroneFrame with .frame, .calibration, .drone_id
print(frame.drone_id, frame.frame.shape, frame.calibration.K[0, 0])

receiver.stop()
```

For all 8 drones:

```python
from algorithm.src.ingestion.enet_receiver import (
    create_enet_receivers, start_all_enet_receivers, stop_all_enet_receivers
)

receivers = create_enet_receivers(list(range(1, 9)), output_queue=q)
start_all_enet_receivers(receivers)
# ... process frames ...
stop_all_enet_receivers(receivers)
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: No module named 'enet'` | pyenet not installed | `pip install pyenet` |
| `FileNotFoundError` on startup | Dataset path wrong | Check `--dataset` path; synthetic fallback activates automatically |
| Receiver gets no frames | Streamer not running | Start streamer before receiver |
| `Connection refused` on receiver | Port mismatch | Verify both sides use same `base_port + drone_id - 1` formula |
| Low frame rate | FPS too high for CPU | Reduce `--fps` or `--jpeg-quality` |
| Packet parse errors in receiver | Format mismatch | Ensure both sides use `<BIQ` + `<9f9f3f5f` struct formats |

---

## Port Reference

| Drone | Default Port |
|-------|-------------|
| 1     | 16000        |
| 2     | 16001        |
| 3     | 16002        |
| 4     | 16003        |
| 5     | 16004        |
| 6     | 16005        |
| 7     | 16006        |
| 8     | 16007        |

The TCP mock_drone_streamer uses ports in the 15000 range, so there is no conflict.
