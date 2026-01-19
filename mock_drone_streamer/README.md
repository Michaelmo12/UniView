# Mock Drone Streamer

TCP streaming server that simulates 8 drones sending frames and calibration data from the MATRIX dataset.

## Overview

- **8 independent drone workers** running in separate threads
- **TCP ports 15000-15007** (one port per drone)
- **2 FPS streaming** with precise timing
- **Binary protocol** for efficient transmission (~200KB per frame)
- **Microservice architecture** - each drone streams independently

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

Server starts all 8 drones and waits for clients to connect. Streams continue even without clients (realistic drone simulation).

## Architecture

```
server.py           Main server - spawns 8 drone worker threads
├── config/         Modular configuration (server, dataset, network, protocol)
├── utils/
│   ├── calibration_loader.py    Load intrinsic/extrinsic calibration
│   ├── frame_loader.py          Load frames from MATRIX dataset
│   └── protocol_encoder.py      Binary packet encoding
└── drone/          (TODO: Modularize worker into classes)
```

## Configuration

All settings in `config/`:
- **ServerConfig** - FPS, frame count, timeouts
- **DatasetConfig** - MATRIX dataset path
- **NetworkConfig** - Port configuration (15000-15007)
- **ProtocolConfig** - Binary protocol constants

See [config/README.md](config/README.md) for details.

## Binary Protocol

Each packet contains:
- **Header** (17 bytes) - drone_id, frame_num, jpeg_size, timestamp
- **JPEG frame** (~200KB) - Compressed image
- **Extrinsic** (48 bytes) - rvec + tvec (raw binary)
- **Intrinsic** (variable) - K matrix + distortion (text)

## Features

- Synchronized startup - all drones start streaming together
- Non-blocking client connections
- Graceful shutdown (Ctrl+C)
- Statistics tracking per drone
- Proper error handling and logging
- No magic numbers - all constants in config

## Dataset

Expects MATRIX_30x30 dataset at:
```
C:\Projects_H.W\FINAL-PROJECT\UniView\MATRIX_30x30\MATRIX_30x30
```

Update path in `config/dataset_config.py` if different.