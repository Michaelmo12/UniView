"""
GlobalIDManager

Monotonically increasing counter that assigns unique global IDs to new tracks.
ID 0 is reserved for "unassigned". IDs start at 1 and are never reused.
"""

import logging

logger = logging.getLogger(__name__)


class GlobalIDManager:
    """
    Assigns unique, monotonically increasing integer IDs to new tracks.

    ID 0 is reserved (unassigned). All issued IDs are >= 1 and never reused,
    even after tracks are retired.
    """

    def __init__(self) -> None:
        self._next_id: int = 1

    def next_id(self) -> int:
        """Return the next unique ID and advance the counter."""
        issued = self._next_id
        self._next_id += 1
        return issued

    def reset(self) -> None:
        """Reset the counter to 1. For testing only."""
        self._next_id = 1

    @property
    def current_count(self) -> int:
        """Total number of IDs issued so far."""
        return self._next_id - 1
