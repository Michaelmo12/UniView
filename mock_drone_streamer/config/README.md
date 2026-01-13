# Mock Drone Streamer Configuration

This folder contains all configuration modules for the mock drone streaming service. Configuration is organized into separate, focused modules following standard modularization practices.

## Configuration Modules

### 1. `server_config.py` - Server Configuration
Server runtime settings and behavior parameters.

**Key Settings:**
- `NUM_DRONES`: Number of drone workers (default: 8)
- `FPS`: Stream rate in frames per second (default: 2)
- `FRAME_INTERVAL`: Calculated frame interval in seconds (0.5s)
- `TOTAL_FRAMES`: Total frames to stream per drone (default: 1000)
- `SOCKET_TIMEOUT`: Non-blocking accept timeout (default: 0.1s)
- `SOCKET_HOST`: Listen address (default: '0.0.0.0')
- `THREAD_JOIN_TIMEOUT`: Thread shutdown timeout (default: 2.0s)
- `PROGRESS_LOG_INTERVAL`: Log interval when client connected (default: 100 frames)
- `NO_CLIENT_LOG_INTERVAL`: Log interval when no client (default: 10 frames)
- `STARTUP_DELAY`: Delay before streaming starts (default: 1.0s)

**Usage:**
```python
from config import ServerConfig

num_drones = ServerConfig.NUM_DRONES
fps = ServerConfig.FPS
```

### 2. `dataset_config.py` - Dataset Configuration
MATRIX dataset path and validation.

**Key Settings:**
- `DATASET_PATH`: Path to MATRIX_30x30 dataset

**Methods:**
- `validate_path()`: Check if dataset exists
- `get_path()`: Get dataset path as Path object

**Usage:**
```python
from config import DatasetConfig

if DatasetConfig.validate_path():
    dataset = DatasetConfig.get_path()
```

### 3. `network_config.py` - Network Configuration
TCP networking settings (ports, addresses).

**Key Settings:**
- `BASE_PORT`: Base TCP port (default: 15000)

**Methods:**
- `get_port(drone_id)`: Get port for specific drone
- `get_port_range(num_drones)`: Get port range string

**Port Mapping:**
```
Drone 1 -> Port 15000
Drone 2 -> Port 15001
...
Drone 8 -> Port 15007
```

**Usage:**
```python
from config import NetworkConfig

port = NetworkConfig.get_port(drone_id=1)  # Returns 15000
range_str = NetworkConfig.get_port_range(8)  # Returns "15000-15007"
```

### 4. `protocol_config.py` - Protocol Configuration
Binary protocol constants for packet encoding/decoding.

**Key Settings:**
- `MIN_DRONE_ID`: Minimum drone ID (default: 1)
- `MAX_DRONE_ID`: Maximum drone ID (default: 8)
- `MIN_FRAME_NUM`: Minimum frame number (default: 0)
- `MAX_FRAME_NUM`: Maximum frame number (default: 999)
- `RVEC_BINARY_SIZE`: Rotation vector size in bytes (24 bytes)
- `TVEC_BINARY_SIZE`: Translation vector size in bytes (24 bytes)
- `EXTRINSIC_BINARY_SIZE`: Total extrinsic size (48 bytes)
- `JPEG_QUALITY`: JPEG compression quality (default: 85)

**Usage:**
```python
from config import ProtocolConfig

if ProtocolConfig.MIN_DRONE_ID <= drone_id <= ProtocolConfig.MAX_DRONE_ID:
    # Valid drone ID
    pass
```

## Import Patterns

### Import All Configs
```python
from config import ServerConfig, DatasetConfig, NetworkConfig, ProtocolConfig
```

### Import Specific Config
```python
from config import ServerConfig
```

### Import from Submodule
```python
from config.server_config import ServerConfig
```

## Modifying Configuration

To change configuration values:

1. Open the appropriate config module
2. Modify the class attribute
3. Changes apply across all imports automatically

**Example:**
```python
# config/server_config.py
class ServerConfig:
    NUM_DRONES = 16  # Changed from 8 to 16
    FPS = 5  # Changed from 2 to 5
```

## Benefits of This Structure

1. **Separation of Concerns**: Each config module handles a specific aspect
2. **Easy to Find**: Related settings grouped logically
3. **Type Safety**: Class attributes provide better IDE support
4. **Extensibility**: Easy to add new config modules
5. **Documentation**: Each module self-documents its purpose
6. **Testability**: Easy to mock configuration in tests
7. **No Magic Numbers**: All constants centralized and named

## File Structure

```
config/
├── __init__.py              # Package exports
├── README.md                # This file
├── server_config.py         # Server runtime settings
├── dataset_config.py        # Dataset path and validation
├── network_config.py        # Network settings (ports)
└── protocol_config.py       # Binary protocol constants
```
