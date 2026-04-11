import logging
from typing import Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class HistoryAggregator:
    """
    Buffers incoming StreamPayloads in memory, grouped by truncated minute.
    When a new minute arrives, flushes the previous minute's aggregate to
    the backend via POST /history/ingest.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, list[dict]] = {}
        self._current_minute: Optional[str] = None

    async def process_payload(self, payload: dict) -> None:
        """
        Buffer a StreamPayload dict by its truncated minute.
        When the minute rolls over, flush the completed minute to the backend.
        """
        timestamp_str: str = payload["timestamp"]
        minute_key: str = timestamp_str[:16]  # e.g. "2026-03-26T14:05"

        if self._current_minute is None:
            self._current_minute = minute_key

        # If the payload belongs to a new minute, flush the previous minute's buffer
        if minute_key != self._current_minute:
            # Previous minute is complete — flush it
            await self._flush(self._current_minute)
            del self._buffers[self._current_minute]
            self._current_minute = minute_key

        # Buffer the payload for the current minute
        self._buffers.setdefault(minute_key, []).append(payload)

    async def _flush(self, minute_key: str) -> None:
        """
        Compute per-minute aggregates from buffered payloads and POST to backend.
        Failures are logged as warnings but never raised.
        """
        payloads = self._buffers.get(minute_key, [])
        if not payloads:
            logger.warning("_flush called for minute %s but buffer is empty", minute_key)
            return

        track_counts = [len(p["tracks"]) for p in payloads]
        avg_people_count = round(sum(track_counts) / len(track_counts))
        peak_people_count = max(track_counts)
        active_drones_count = max(p["active_drones_count"] for p in payloads)
        total_reid_matches = max(p["total_reid_matches"] for p in payloads)

        iso_timestamp = minute_key + ":00+00:00"

        body = {
            "timestamp": iso_timestamp,
            "avg_people_count": avg_people_count,
            "peak_people_count": peak_people_count,
            "active_drones_count": active_drones_count,
            "total_reid_matches": total_reid_matches,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.BACKEND_URL}/history/ingest",
                    json=body,
                    timeout=5.0,
                )
            if response.status_code in (200, 201):
                logger.info(
                    "Flushed history for minute %s (%d payloads)", minute_key, len(payloads)
                )
            else:
                logger.warning(
                    "Backend returned %s when ingesting history for minute %s",
                    response.status_code,
                    minute_key,
                )
        except Exception as exc:
            logger.warning(
                "Failed to flush history for minute %s: %s", minute_key, exc
            )


    def get_current_status(self) -> dict:
        """
        Return the current in-memory system status derived from the active minute's buffer.
        Called by GET /api/algorithm/status to give the frontend a snapshot.
        """
        _default = {"active_drones": 0, "active_tracks": 0, "server_fps": 0.0, "system_status": "Optimal"}

        if self._current_minute is None:
            return _default

        payloads = self._buffers.get(self._current_minute, [])
        if not payloads:
            return _default

        # active_drones: max reported across all payloads this minute
        active_drones = max(p.get("active_drones_count", 0) for p in payloads)

        # active_tracks: unique global_ids from the *latest* payload per drone
        # (avoids double-counting the same person seen by multiple drones)
        latest_per_drone: dict[str, dict] = {}
        for p in payloads:
            drone_id = p.get("drone_id", "")
            latest_per_drone[drone_id] = p  # later payloads overwrite earlier ones

        unique_global_ids: set[int] = set()
        for p in latest_per_drone.values():
            for track in p.get("tracks", []):
                gid = track.get("global_id")
                if gid is not None:
                    unique_global_ids.add(gid)
        active_tracks = len(unique_global_ids)

        # avg_confidence across all payloads this minute
        confidences = [p.get("avg_confidence", 0.0) for p in payloads if p.get("avg_confidence", 0.0) > 0]
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        # avg_latency across all payloads this minute
        latencies = [p.get("pipeline_latency_ms", 0.0) for p in payloads]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        server_fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

        if avg_latency < 1000:
            system_status = "Optimal"
        elif avg_latency < 2000:
            system_status = "Warning"
        else:
            system_status = "Critical"

        # avg per-stage timings from payloads that include them
        stage_keys = ("detection", "features", "fusion", "reconstruction", "tracking", "total")
        stage_samples: dict[str, list[float]] = {k: [] for k in stage_keys}
        for p in payloads:
            timings = p.get("stage_timings_ms", {})
            for k in stage_keys:
                if k in timings:
                    stage_samples[k].append(timings[k])
        avg_stage_timings = {
            k: round(sum(v) / len(v), 1) if v else 0.0
            for k, v in stage_samples.items()
        }

        return {
            "active_drones": active_drones,
            "active_tracks": active_tracks,
            "server_fps": server_fps,
            "system_status": system_status,
            "avg_pipeline_latency_ms": round(avg_latency, 1),
            "avg_confidence": avg_confidence,
            "stage_timings_ms": avg_stage_timings,
        }


# Module-level singleton used by the push handler
history_aggregator = HistoryAggregator()
