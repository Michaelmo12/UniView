"""
API Schemas

Pydantic models describing the structure of WebSocket messages sent to frontend clients.
These are for documentation and validation purposes only — the pipeline serializes
directly to dicts via output_formatter.py.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class DetectionInfo(BaseModel):
    """Per-camera detection linked to a tracked person."""

    drone_id: int
    local_id: int
    confidence: Optional[float] = None
    bbox: Optional[Dict[str, float]] = None  # {"x1", "y1", "x2", "y2"}


class TrackedPersonMsg(BaseModel):
    """A confirmed tracked person with Kalman-smoothed position."""

    global_id: int
    position: List[float]          # [x, y, z] in meters — plain Python list
    velocity: List[float]          # [vx, vy, vz] in m/frame
    state: str                     # "CONFIRMED"
    frames_tracked: int
    frames_since_update: int
    detections: List[DetectionInfo]


class FrameMessage(BaseModel):
    """
    One WebSocket broadcast message sent per processed synchronized frame set.

    frames maps drone_id (as string) to base64 JPEG data URL.
    No bounding boxes are drawn on the frames.
    """

    frame_num: int
    timestamp: float
    frames: Dict[str, str]         # {drone_id_str: "data:image/jpeg;base64,..."}
    tracked_persons: List[TrackedPersonMsg]
