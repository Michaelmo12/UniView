# ENet Drone Streamer — Code Explainer

This document explains exactly how the streamer and receiver work, file by file, line by line where it matters.

---
# best combination cameras 3,4,6,7

## Big Picture

```
MATRIX Dataset (PNGs + XML calibration files)
        │
        ▼
  DatasetLoader          reads frames + calibration from disk
        │
        ▼
  PacketBuilder          packs everything into one binary blob
        │
        ▼
  ENetStreamer           sends that blob over UDP to anyone connected
        │
      (network — UDP via ENet)
        │
        ▼
  ENetReceiver           receives the blob, unpacks it
        │
        ▼
  DroneFrame object      frame (numpy BGR image) + CameraCalibration
        │
        ▼
  algorithm pipeline     detection → fusion → tracking → output
```

Each drone runs its own independent streamer process on its own port (16000–16007).

---

## Streamer Side

### `config/config.py` — StreamerConfig

A Python dataclass that holds all settings for one streamer instance.

```python
@dataclass
class StreamerConfig:
    dataset_path: str   # path to MATRIX_30x30/MATRIX_30x30/
    drone_id: int       # 1–8
    host: str           # "0.0.0.0" = accept from any IP
    base_port: int      # 16000
    fps: float          # 2.0
    jpeg_quality: int   # 85
    loop: bool          # True = restart from frame 0 after frame 999
    num_drones: int     # 8

    @property
    def port(self) -> int:
        return self.base_port + (self.drone_id - 1)
        # Drone 1 → 16000, Drone 2 → 16001, ..., Drone 8 → 16007
```

`port` is a computed property — it's not stored, it's calculated from `base_port + drone_id - 1` every time you access it.

---

### `src/dataset_loader.py` — DatasetLoader

Reads PNG frames and XML calibration files from the MATRIX dataset for one specific drone.

#### `__init__`

```python
def __init__(self, dataset_path, drone_id, jpeg_quality):
    self._frames_dir = dataset_path / "image_subsets" / f"D{drone_id}"
    self._extr_dir   = dataset_path / "calibrations" / "extrinsic"
    self._intr_dir   = dataset_path / "calibrations" / "intrinsic"
```

Builds the three paths it needs. Then checks all three exist — if any is missing it raises `FileNotFoundError` immediately so you know right away rather than failing silently at frame 0.

#### `get_frame_count`

Globs `[0-9][0-9][0-9][0-9].png` in the frames directory and counts them. Returns 1000 for a full MATRIX dataset (frames 0000–0999).

#### `load_frame(frame_idx)`

Main method. Called by the streamer for every frame tick.

Returns a 5-tuple: `(jpeg_bytes, K, R, t, dist)`

```
jpeg_bytes  — the PNG read from disk, re-encoded as JPEG at the configured quality
K           — (3,3) float32 intrinsic matrix
R           — (3,3) float32 rotation matrix  ← converted from rvec
t           — (3,1) float32 translation vector
dist        — (5,)  float32 distortion coefficients
```

#### `_load_extrinsic(frame_idx)`

Reads `calibrations/extrinsic/extr_Drone{N}_{frame:04d}.xml`.

The MATRIX dataset stores rotation as a **rotation vector** (rvec) — 3 numbers that compactly represent an axis of rotation and the angle around it. This is OpenCV's Rodrigues format. It is NOT a 3×3 matrix.

```python
rvec_binary = self._extract_binary_element(root, "rvec")  # base64 decoded
rvec_vals = struct.unpack("ddd", rvec_binary[-24:])        # 3 float64 values
rvec = np.array(rvec_vals).reshape(3, 1)

R, _ = cv2.Rodrigues(rvec)   # converts 3-number rvec → 3×3 rotation matrix
R = R.astype(np.float32)
```

Why `[-24:]` (last 24 bytes)? The XML binary blob has a prefix before the actual double values. 3 × 8 bytes = 24 bytes, so taking the last 24 always gives the correct values regardless of prefix length — same technique used in the TCP streamer.

#### `_load_intrinsic(frame_idx)`

Reads `calibrations/intrinsic/intr_Drone{N}_{frame:04d}.xml`.

Parses the `<camera_matrix>` element as space-separated floats → reshapes to (3,3).
Parses the `<distortion_coefficients>` element as space-separated floats → takes first 5.

The XML stores these as `float64` (`dt>d</dt>`). We cast to `float32` when building the numpy arrays — tiny precision loss, irrelevant for camera math.

#### `_extract_binary_element`

```python
binary_data = base64.b64decode(data_elem.text.strip())
```

The MATRIX XML stores rvec/tvec as base64-encoded raw binary (OpenCV's binary XML format). This decodes that base64 back to raw bytes, then the caller unpacks the doubles.

---

### `src/packet_builder.py` — PacketBuilder

Assembles the binary packet that gets sent over the network.

#### Packet layout

```
Offset   Size    Type        Field
──────────────────────────────────────────────────────
0        1       uint8       drone_id
1        4       uint32 LE   frame_num
5        8       uint64 LE   timestamp_ns
──────── header = 13 bytes ─────────────────────────
13       36      9×float32   K matrix (row-major)
49       36      9×float32   R matrix (row-major)
85       12      3×float32   t vector
97       20      5×float32   dist coefficients
──────── calibration = 104 bytes ───────────────────
117      N       bytes       JPEG image data
──────── total fixed = 117 bytes + JPEG ────────────
```

#### `build_packet`

```python
header = struct.pack("<BIQ", drone_id, frame_num, timestamp_ns)
#                     │ │ └─ uint64 (8 bytes) — nanoseconds since epoch
#                     │ └─── uint32 (4 bytes) — frame sequence number
#                     └───── uint8  (1 byte)  — drone ID
```

`<` means little-endian. `B` = unsigned byte, `I` = unsigned int, `Q` = unsigned long long.

```python
calibration = struct.pack("<9f9f3f5f",
    *K.flatten(),   # 9 floats = row 0, row 1, row 2 of K
    *R.flatten(),   # 9 floats = row 0, row 1, row 2 of R
    *t.flatten(),   # 3 floats = tx, ty, tz
    *dist.flatten() # 5 floats = k1, k2, p1, p2, k3
)
```

Then simply concatenates: `header + calibration + jpeg_bytes`. No length prefix for the JPEG — the receiver knows JPEG starts at byte 117 and runs to the end of the packet.

---

### `src/streamer.py` — ENetStreamer

The ENet server. Runs a blocking loop, accepts connections, sends frames.

#### `__init__`

```python
self._loader = DatasetLoader(...)   # reads from disk
self._builder = PacketBuilder()     # builds binary packets
self._peers = []                    # list of currently connected ENet peers
self._frame_count = ...             # 1000 for MATRIX
self._current_frame_idx = 0
self._frames_sent = 0
self._total_bytes_sent = 0
```

#### `run()`

Creates the ENet host (server socket):

```python
host = enet.Host(
    enet.Address(b"0.0.0.0", port),  # bind to all interfaces on this port
    peerCount=10,                    # accept up to 10 simultaneous clients
    channelLimit=1,                  # only one channel needed
    incomingBandwidth=0,             # 0 = unlimited
    outgoingBandwidth=0,             # 0 = unlimited
)
```

Then enters the main loop:

```python
while True:
    event = host.service(10)     # poll ENet for up to 10ms
    self._handle_event(event)    # handle connect/disconnect/receive

    now = time.monotonic()
    if now - last_send_time >= frame_interval:   # 0.5s at 2 FPS
        self._send_frame()
        last_send_time = now
```

`host.service(10)` does two things: processes incoming network events (connect/disconnect/receive) AND flushes outgoing packets. The 10ms timeout means the loop runs approximately 100 times per second, but only sends a frame every 500ms.

#### `_handle_event`

Three event types:
- `EVENT_TYPE_CONNECT` — a new client connected, add to `self._peers`
- `EVENT_TYPE_DISCONNECT` — a client left, remove from `self._peers`
- `EVENT_TYPE_RECEIVE` — unexpected inbound data, just log it (streamer doesn't expect to receive anything)

#### `_send_frame`

```python
if not self._peers:
    # No one is connected — still advance the frame counter, don't build packet
    if frame_idx % 10 == 0:
        print(f"[DRONE {id}] Processing frame {frame_idx:04d} (no clients connected)")
    self._advance_frame()
    return
```

If someone IS connected:
1. `self._loader.load_frame(frame_idx)` → gets `(jpeg_bytes, K, R, t, dist)`
2. `self._builder.build_packet(...)` → assembles binary blob
3. `enet.Packet(packet_data, enet.PACKET_FLAG_RELIABLE)` → wraps it with reliability flag
4. `peer.send(0, packet)` → sends on channel 0 to each peer

`PACKET_FLAG_RELIABLE` means ENet will retransmit if the packet is dropped, acknowledge delivery, and maintain ordering — same guarantees as TCP but over UDP.

Progress logged every 100 frames when clients connected.

#### `_advance_frame`

Increments `_current_frame_idx`. When it hits 1000 (end of dataset):
- If `loop=True` → reset to 0 (default behavior)
- If `loop=False` → raise `StopIteration` → process exits

---

### `main.py` — Entry point

Parses CLI args, builds a `StreamerConfig`, creates an `ENetStreamer`, calls `run()`.

The only tricky part:

```python
base_port = (args.port - (args.drone_id - 1)) if args.port is not None else 16000
```

If you pass `--port 16003`, it back-calculates `base_port` so that `config.port` (which does `base_port + drone_id - 1`) returns 16003. This lets the `--port` flag override the computed value.

---

### `run_all_drones.py` — Multi-process launcher

Spawns 8 independent `main.py` subprocesses, one per drone.

Key detail:
```python
env = os.environ.copy()
env["PYTHONPATH"] = str(script_dir.parent)  # adds UniView/ to path
proc = subprocess.Popen(cmd, env=env)
```

Each subprocess needs `UniView/` on its Python path so `from enet_drone_streamer.config.config import ...` resolves correctly. Without this you get `ModuleNotFoundError`.

After spawning all 8:
```python
time.sleep(1.0)  # wait 1 second — synchronized start
```

This gives all 8 processes time to initialize before any start streaming, so they all begin at roughly frame 0 at the same time.

On `Ctrl+C`: sends `SIGTERM` to all 8 child processes, then waits up to 5 seconds for each to exit cleanly. Any that don't exit get `SIGKILL`.

---

## Receiver Side

### `algorithm/src/ingestion/enet_receiver.py` — ENetReceiver

Connects to one drone streamer as an ENet **client** and pushes decoded `DroneFrame` objects onto a queue.

#### Public API (matches TCPReceiver)

```python
receiver = ENetReceiver(drone_id=1, output_queue=q)
receiver.start()          # spawns background thread
frame = q.get(timeout=5)  # DroneFrame object
receiver.stop()           # signals thread to exit
```

Same interface as `TCPReceiver` — drop-in replacement.

#### `__init__`

```python
self.port = base_port + (drone_id - 1)   # mirrors streamer port calculation
self._stop_event = threading.Event()      # used to signal thread shutdown
self._running = False
self._connected = False
```

#### `start()`

Spawns a daemon thread running `_run()`. Daemon threads die automatically when the main process exits — no dangling threads if something crashes.

#### `_run()` — the receive loop

Outer loop: keeps reconnecting until `stop()` is called.

```python
while not self._stop_event.is_set():
    host = enet.Host(None, peerCount=1, ...)   # None = client mode (no bind address)
    peer = host.connect(enet.Address(host_bytes, port), 1)

    # Inner loop: process events while connected
    while not self._stop_event.is_set():
        event = host.service(100)   # poll for up to 100ms

        if event.type == EVENT_TYPE_CONNECT:
            self._connected = True

        elif event.type == EVENT_TYPE_DISCONNECT:
            self._connected = False
            break   # exit inner loop → outer loop will reconnect

        elif event.type == EVENT_TYPE_RECEIVE:
            data = bytes(event.packet.data)
            frame = self._parse_packet(data)
            self.output_queue.put(frame)
```

When the streamer goes down, `EVENT_TYPE_DISCONNECT` fires, inner loop breaks, outer loop waits `reconnect_delay` seconds (default 5s) then tries to reconnect.

#### `_parse_packet(data)` — the inverse of PacketBuilder

```python
# Header — 13 bytes
drone_id, frame_num, timestamp_ns = struct.unpack("<BIQ", data[:13])

# Calibration — 104 bytes
calib_values = struct.unpack("<9f9f3f5f", data[13:117])
K    = np.array(calib_values[0:9],  dtype=np.float32).reshape(3, 3)
R    = np.array(calib_values[9:18], dtype=np.float32).reshape(3, 3)
t    = np.array(calib_values[18:21],dtype=np.float32).reshape(3, 1)
dist = np.array(calib_values[21:26],dtype=np.float32)

# JPEG — everything from byte 117 to end
jpeg_bytes = data[117:]
image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

# timestamp: nanoseconds → seconds
timestamp_sec = timestamp_ns / 1e9

# Assemble output
calib = CameraCalibration(K=K, R=R, t=t, dist=dist)
frame = DroneFrame(drone_id=..., frame_num=..., timestamp=timestamp_sec,
                   frame=image, calibration=calib)
```

Exact mirror of `PacketBuilder.build_packet`. Same byte offsets, same format strings, same endianness.

---

## Packet Format — Full Reference

```
Byte offset   Size    Struct   Value
────────────────────────────────────────────────────────
0             1       B        drone_id         (1–8)
1             4       I        frame_num        (0–999)
5             8       Q        timestamp_ns     (uint64, nanoseconds)
────────────────────────────────────────────────────────
13            4       f        K[0,0] = fx
17            4       f        K[0,1] = 0
21            4       f        K[0,2] = cx
25            4       f        K[1,0] = 0
29            4       f        K[1,1] = fy
33            4       f        K[1,2] = cy
37            4       f        K[2,0] = 0
41            4       f        K[2,1] = 0
45            4       f        K[2,2] = 1
────────────────────────────────────────────────────────
49            36      9f       R matrix, row-major (R[0,0]...R[2,2])
────────────────────────────────────────────────────────
85            4       f        t[0] = tx
89            4       f        t[1] = ty
93            4       f        t[2] = tz
────────────────────────────────────────────────────────
97            4       f        dist[0] = k1
101           4       f        dist[1] = k2
105           4       f        dist[2] = p1
109           4       f        dist[3] = p2
113           4       f        dist[4] = k3
────────────────────────────────────────────────────────
117           N       bytes    JPEG image data
────────────────────────────────────────────────────────
Total fixed header: 117 bytes
Total packet: ~200–400 KB per frame (1920×1080 JPEG at quality 85)
```

All multi-byte values are **little-endian**.

---

## Why ENet Instead of TCP

| Property | TCP | ENet (UDP) |
|---|---|---|
| Transport | TCP | UDP |
| Reliability | Built-in | Optional per-packet (`PACKET_FLAG_RELIABLE`) |
| Fragmentation | OS handles it | ENet handles it automatically |
| Large frames (>65KB) | Works, but needs length-prefix framing | Works natively |
| Latency | Higher (Nagle, ACK round-trips) | Lower |
| Packet format | Stream — must manually delimit packets | Message-based — each send = one receive |

With TCP you have to manually prefix every packet with its length so the receiver knows where one packet ends and the next begins. With ENet each `peer.send()` corresponds to exactly one `EVENT_TYPE_RECEIVE` on the other side — no framing needed.

---

## Running

```bash
# All 8 drones (from UniView/)
cd enet_drone_streamer
python run_all_drones.py

# Single drone
cd UniView/
python enet_drone_streamer/main.py --drone-id 1

# Test receiver (live, requires streamer running)
cd UniView/
python algorithm/src/ingestion/test_enet_receiver.py --drone-id 1 --frames 5

# Offline unit test (no streamer needed)
python algorithm/src/ingestion/test_enet_receiver.py --unit-test
```
