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

        if minute_key != self._current_minute:
            # Previous minute is complete — flush it
            await self._flush(self._current_minute)
            del self._buffers[self._current_minute]
            self._current_minute = minute_key

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
        avg_people_count = sum(track_counts) / len(track_counts)
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


# Module-level singleton used by the push handler
history_aggregator = HistoryAggregator()
