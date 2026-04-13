"""
HistoryLog model - persists aggregated per-minute tracking statistics
Written by the gateway aggregator, read by the history dashboard frontend.
"""
from sqlalchemy import Column, Integer, Float, DateTime
from src.core.database import Base


class HistoryLog(Base):
    """
    HistoryLog model - represents the 'history_logs' table in PostgreSQL

    Each row captures one minute of aggregated tracking data.
    The timestamp column is unique (one row per minute).
    """
    __tablename__ = "history_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), unique=True, index=True, nullable=False)
    avg_people_count = Column(Integer, nullable=False)
    peak_people_count = Column(Integer, nullable=False)
    active_drones_count = Column(Integer, nullable=False)
    total_reid_matches = Column(Integer, nullable=False)

    def __repr__(self):
        return (
            f"<HistoryLog(id={self.id}, timestamp={self.timestamp}, "
            f"avg_people={self.avg_people_count}, peak={self.peak_people_count})>"
        )
