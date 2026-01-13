"""
Server Configuration

All server-related configuration settings for the mock drone streamer service.
"""


class ServerConfig:
    """Server configuration constants"""

    # Number of drone workers to spawn
    NUM_DRONES = 8

    # Stream rate (frames per second)
    FPS = 2

    # Calculated frame interval (seconds)
    FRAME_INTERVAL = 1.0 / FPS  # 0.5 seconds

    # Total frames to stream per drone
    TOTAL_FRAMES = 1000

    # Socket settings
    SOCKET_TIMEOUT = 0.1  # Non-blocking accept timeout (seconds)
    SOCKET_HOST = '0.0.0.0'  # Listen on all interfaces

    # Thread settings
    THREAD_JOIN_TIMEOUT = 2.0  # Seconds to wait for thread shutdown

    # Logging settings
    PROGRESS_LOG_INTERVAL = 100  # Log progress every N frames (when client connected)
    NO_CLIENT_LOG_INTERVAL = 10  # Log progress every N frames (when no client)

    # Synchronization
    STARTUP_DELAY = 1.0  # Seconds to wait before starting stream after all drones ready
