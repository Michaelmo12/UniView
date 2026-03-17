"""
Output Formatter

Converts pipeline stage outputs (TrackingResult + SynchronizedFrameSet + detection_sets)
into a JSON-serializable dict suitable for WebSocket broadcast.

Rules:
- No bounding boxes drawn on frames (raw JPEG only)
- All numpy arrays converted to plain Python types
- Raw JPEG encoding via cv2.imencode
- Frames encoded as base64 data URLs
"""

import base64
import json
import logging
from typing import Dict

import cv2
import numpy as np

from algorithm.ingestion.models import SynchronizedFrameSet
from algorithm.detection.models import DetectionSet
from algorithm.tracking.models import TrackingResult, TrackedPerson
from algorithm.config.settings import settings

logger = logging.getLogger(__name__)


def format_output(
    tracking_result: TrackingResult,
    sync_set: SynchronizedFrameSet,
    detection_sets: Dict[int, DetectionSet],
) -> str:
    """
    Serialize one frame's pipeline output into a JSON string for WebSocket broadcast.

    Args:
        tracking_result: Output from PersonTracker.update()
        sync_set: SynchronizedFrameSet with raw frames (no bboxes drawn)
        detection_sets: Dict[drone_id -> DetectionSet] for per-camera detection linkage

    Returns:
        JSON string with:
          frame_num: int
          timestamp: float
          frames: {drone_id_str -> "data:image/jpeg;base64,<...>"}
          tracked_persons: list of person dicts
    """
    quality = settings.output.jpeg_quality

    # --- Encode raw frames as base64 JPEG data URLs ---
    frames_dict: dict[str, str] = {}
    for drone_id, drone_frame in sync_set.frames.items():
        success, jpeg_buf = cv2.imencode(
            ".jpg",
            drone_frame.frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if success:
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
            frames_dict[str(drone_id)] = f"data:image/jpeg;base64,{b64}"
        else:
            logger.warning("Failed to encode frame for drone %d", drone_id)

    # --- Build detection lookup: (drone_id, local_id) -> Detection ---
    detection_lookup: dict[tuple[int, int], object] = {}
    for drone_id, det_set in detection_sets.items():
        for det in det_set.detections:
            detection_lookup[(det.drone_id, det.local_id)] = det

    # --- Serialize tracked persons ---
    tracked_persons_list = []
    for tp in tracking_result.tracked_persons:
        person_dict = _serialize_tracked_person(tp, detection_lookup)
        tracked_persons_list.append(person_dict)

    payload = {
        "frame_num": int(tracking_result.frame_num),
        "timestamp": float(sync_set.timestamp),
        "frames": frames_dict,
        "tracked_persons": tracked_persons_list,
    }

    return json.dumps(payload)


def _serialize_tracked_person(
    tp: TrackedPerson,
    detection_lookup: dict,
) -> dict:
    """
    Convert a TrackedPerson to a JSON-serializable dict.

    Returns plain Python types only (no numpy arrays).
    """
    # Convert numpy position/velocity to plain Python lists
    position = [float(v) for v in tp.position.tolist()]
    velocity = [float(v) for v in tp.velocity.tolist()]

    # Build per-camera detection info from source_person.source_detections
    detections_per_camera = []
    if tp.source_person is not None:
        for drone_id, local_id in tp.source_person.source_detections:
            det = detection_lookup.get((drone_id, local_id))
            cam_entry: dict = {
                "drone_id": int(drone_id),
                "local_id": int(local_id),
            }
            if det is not None:
                cam_entry["confidence"] = float(det.confidence)
                cam_entry["bbox"] = {
                    "x1": float(det.bbox.x1),
                    "y1": float(det.bbox.y1),
                    "x2": float(det.bbox.x2),
                    "y2": float(det.bbox.y2),
                }
            detections_per_camera.append(cam_entry)

    return {
        "global_id": int(tp.global_id),
        "position": position,
        "velocity": velocity,
        "state": tp.state.value,
        "frames_tracked": int(tp.frames_tracked),
        "frames_since_update": int(tp.frames_since_update),
        "detections": detections_per_camera,
    }
