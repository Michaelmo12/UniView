"""
History routes — GET /history/ and POST /history/ingest

GET /history/ returns aggregated per-minute tracking rows ordered by timestamp ASC.
POST /history/ingest accepts a completed-minute payload from the gateway aggregator.
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core import get_db
from models import HistoryLog

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HistoryLogResponse(BaseModel):
    id: int
    timestamp: str
    avg_people_count: float
    peak_people_count: int
    active_drones_count: int
    total_reid_matches: int

    class Config:
        from_attributes = True


class HistoryIngestRequest(BaseModel):
    timestamp: str
    avg_people_count: float
    peak_people_count: int
    active_drones_count: int
    total_reid_matches: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(value: str, param_name: str) -> datetime:
    """Parse an ISO 8601 string; raise HTTP 400 on failure."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ISO 8601 datetime for '{param_name}': {value!r}",
        )


def _row_to_dict(row: HistoryLog) -> dict:
    return {
        "id": row.id,
        "timestamp": row.timestamp.isoformat(),
        "avg_people_count": row.avg_people_count,
        "peak_people_count": row.peak_people_count,
        "active_drones_count": row.active_drones_count,
        "total_reid_matches": row.total_reid_matches,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[HistoryLogResponse])
def get_history(
    start_time: Optional[str] = Query(None, description="ISO 8601 start of range (inclusive)"),
    end_time: Optional[str] = Query(None, description="ISO 8601 end of range (inclusive)"),
    db: Session = Depends(get_db),
):
    """
    Return history log rows ordered by timestamp ASC.

    Optional query parameters:
    - start_time: ISO 8601 string — only rows with timestamp >= start_time
    - end_time:   ISO 8601 string — only rows with timestamp <= end_time
    """
    query = db.query(HistoryLog)

    if start_time is not None:
        dt_start = _parse_iso(start_time, "start_time")
        query = query.filter(HistoryLog.timestamp >= dt_start)

    if end_time is not None:
        dt_end = _parse_iso(end_time, "end_time")
        query = query.filter(HistoryLog.timestamp <= dt_end)

    rows = query.order_by(HistoryLog.timestamp.asc()).all()
    return [_row_to_dict(r) for r in rows]


@router.post("/ingest", status_code=status.HTTP_201_CREATED, response_model=HistoryLogResponse)
def ingest_history(
    payload: HistoryIngestRequest,
    db: Session = Depends(get_db),
):
    """
    Accept an aggregated minute payload from the gateway aggregator and persist it.

    If a row for the given timestamp already exists it is updated (upsert).
    No authentication required — this endpoint is internal-network only.
    """
    ts = _parse_iso(payload.timestamp, "timestamp")

    # Check for existing row on this timestamp (upsert pattern)
    existing = db.query(HistoryLog).filter(HistoryLog.timestamp == ts).first()

    if existing:
        existing.avg_people_count = payload.avg_people_count
        existing.peak_people_count = payload.peak_people_count
        existing.active_drones_count = payload.active_drones_count
        existing.total_reid_matches = payload.total_reid_matches
        db.commit()
        db.refresh(existing)
        logger.info(f"Updated HistoryLog for timestamp={ts.isoformat()}")
        return _row_to_dict(existing)

    new_row = HistoryLog(
        timestamp=ts,
        avg_people_count=payload.avg_people_count,
        peak_people_count=payload.peak_people_count,
        active_drones_count=payload.active_drones_count,
        total_reid_matches=payload.total_reid_matches,
    )

    try:
        db.add(new_row)
        db.commit()
        db.refresh(new_row)
    except IntegrityError:
        db.rollback()
        # Race condition — row was inserted between our query and commit; update it
        existing = db.query(HistoryLog).filter(HistoryLog.timestamp == ts).first()
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to insert or find HistoryLog row",
            )
        existing.avg_people_count = payload.avg_people_count
        existing.peak_people_count = payload.peak_people_count
        existing.active_drones_count = payload.active_drones_count
        existing.total_reid_matches = payload.total_reid_matches
        db.commit()
        db.refresh(existing)
        return _row_to_dict(existing)

    logger.info(f"Inserted HistoryLog for timestamp={ts.isoformat()}")
    return _row_to_dict(new_row)
