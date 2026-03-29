"""
Output Formatter

Builds StreamPayload dicts (one per drone) from TrackingResult + SynchronizedFrameSet.
Called once per frame in the pipeline loop; yields one dict per active drone.

StreamPayload schema (mirrors gateway's StreamPayload Pydantic model):
  timestamp          — ISO-8601 UTC string
  drone_id           — str (e.g. "1")
  frame_base64       — raw base64 JPEG, no data: prefix (frontend adds it)
  tracks             — CONFIRMED tracks visible on this drone (list[TrackEntry])
  active_drones_count
  total_reid_matches — number of cross-camera matches this frame
  pipeline_latency_ms
  avg_confidence

Rules:
- NO bboxes drawn on frames. Raw JPEG only.
- All numpy types converted to plain Python primitives.
- Only CONFIRMED tracked persons emitted (TrackedPerson.state == CONFIRMED).
- Drones with no frame data are skipped silently.
"""
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Iterator

import cv2

from src.config.settings import settings
from src.detection.models import DetectionSet
from src.ingestion.models import SynchronizedFrameSet
from src.tracking.models import TrackingResult

logger = logging.getLogger(__name__)


def build_payloads(
    result: TrackingResult,
    detection_sets: dict[int, DetectionSet],
    sync_set: SynchronizedFrameSet,
    pipeline_start_time: float,
    stage_timings_ms: dict | None = None,
) -> Iterator[dict]:
    """
    Yield one StreamPayload dict per active drone in sync_set.

    Args:
        result: TrackingResult from PersonTracker.update()
        detection_sets: {drone_id: DetectionSet} from detection stage
        sync_set: SynchronizedFrameSet containing raw frames
        pipeline_start_time: time.monotonic() captured before pipeline stages ran
    """
    quality = settings.output.jpeg_quality
    latency_ms = (time.monotonic() - pipeline_start_time) * 1000
    timestamp = datetime.now(timezone.utc).isoformat()
    active_drones_count = len(sync_set.frames)

    # Count cross-camera matches from this frame
    total_reid_matches = len(result.tracked_persons)

    # Build detection lookup: (drone_id, local_id) -> Detection
    detection_lookup: dict[tuple[int, int], object] = {}
    for drone_id, det_set in detection_sets.items():
        for det in det_set.detections:
            if det.local_id is not None:
                detection_lookup[(drone_id, det.local_id)] = det

    for drone_id, drone_frame in sync_set.frames.items():
        # Encode raw JPEG — no bboxes drawn
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ok, buf = cv2.imencode(".jpg", drone_frame.frame, encode_params)
        if not ok:
            logger.warning("JPEG encoding failed for drone %d", drone_id)
            continue

        frame_base64 = base64.b64encode(buf.tobytes()).decode("ascii")

        # Build tracks for this drone (CONFIRMED only)
        tracks: list[dict] = []
        confidences: list[float] = []

        for p in result.tracked_persons:
            if p.source_person is None:
                continue

            # Find if this confirmed person has a detection from this drone
            drone_dets = [
                (did, lid)
                for did, lid in p.source_person.source_detections
                if did == drone_id
            ]
            if not drone_dets:
                continue

            _, local_id = drone_dets[0]
            det = detection_lookup.get((drone_id, local_id))

            if det is not None:
                x1, y1, x2, y2 = (
                    float(det.bbox.x1),
                    float(det.bbox.y1),
                    float(det.bbox.x2),
                    float(det.bbox.y2),
                )
                conf = float(det.confidence)
                confidences.append(conf)
            else:
                x1 = y1 = x2 = y2 = 0.0
                conf = 0.0

            tracks.append({
                "global_id": int(p.global_id),
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "confidence": conf,
                "state": p.state.value,
                "frames_tracked": int(p.frames_tracked),
            })

        # Also include single-view persons visible on this drone (global_id = -1)
        for person in result.single_view_persons:
            drone_dets = [
                (did, lid)
                for did, lid in person.source_detections
                if did == drone_id
            ]
            if not drone_dets:
                continue

            _, local_id = drone_dets[0]
            det = detection_lookup.get((drone_id, local_id))
            if det is None:
                continue

            tracks.append({
                "global_id": -1,
                "x": int(float(det.bbox.x1)),
                "y": int(float(det.bbox.y1)),
                "width": int(float(det.bbox.x2) - float(det.bbox.x1)),
                "height": int(float(det.bbox.y2) - float(det.bbox.y1)),
                "confidence": float(det.confidence),
                "state": "SINGLE_VIEW",
                "frames_tracked": 0,
            })

        avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

        yield {
            "timestamp": timestamp,
            "drone_id": str(drone_id),
            "frame_base64": frame_base64,
            "tracks": tracks,
            "active_drones_count": active_drones_count,
            "total_reid_matches": total_reid_matches,
            "pipeline_latency_ms": float(latency_ms),
            "avg_confidence": avg_confidence,
            "stage_timings_ms": stage_timings_ms or {},
        }
